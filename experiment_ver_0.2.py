import dataclasses
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Self, assert_never

import mlflow
import torch
import torch.fx.graph
import torch.nn as nn
from graphene.utils.dataloader import DataLoader
from nn_lib.datasets import ImageNetDataModule, TorchvisionDataModuleBase
from nn_lib.models import get_pretrained_model
from nn_lib.models.graph_module_plus import GraphModulePlus
from nn_lib.models.utils import frozen
from nn_lib.utils import save_as_artifact, search_runs_by_params
from torchmetrics import Accuracy
from tqdm.auto import tqdm

from stitching import create_stitching_layer


def _get_pretrained_model_by_name(model_name: str) -> GraphModulePlus:
    return GraphModulePlus.new_from_trace(
        get_pretrained_model(model_name)
    ).squash_all_conv_batchnorm_pairs()


def _get_dataset_by_name(dataset_name: str) -> TorchvisionDataModuleBase:
    # TODO - implement others and move to an nn_lib registry
    match dataset_name.lower():
        case "imagenet":
            # Imagenet has about 1.2M training images, which we'll split so that we get about 5k
            # validation images. Warning: no checks are done to ensure that the split is balanced.
            dm = ImageNetDataModule(root_dir="/data/datasets/", train_val_split=0.99)
            dm.prepare_data()
            return dm
        case _:
            assert_never(dataset_name)


class TargetType(Enum):
    TASK = auto()
    MATCH_UPSTREAM = auto()
    MATCH_DOWNSTREAM = auto()


@dataclasses.dataclass
class DonorSpec:
    model: str | GraphModulePlus
    layer: str
    dataset: str | TorchvisionDataModuleBase

    def maybe_initialize(self) -> Self:
        if isinstance(self.model, str):
            self.model = _get_pretrained_model_by_name(self.model)
        if isinstance(self.dataset, str):
            self.dataset = _get_dataset_by_name(self.dataset)
        return self


def create_hybrid_model(
    donorA: DonorSpec, donorB: DonorSpec, stitch_family: str
) -> tuple[GraphModulePlus, GraphModulePlus, GraphModulePlus]:
    """Initializes a stitched model, inferring the shapes of the layers to be stitched.

    :param donorA: A DonorSpec object representing the upstream model & layer to be stitched.
    :param donorB: A DonorSpec object representing the downstream model & layer to be stitched.
    :param stitch_family: A string passed to create_stitching_layer specifying the type of
        stitching layer.
    :return: A tuple of
        - modelAxB: The stitched model.
        - donorA_embedding_getter: A GraphModulePlus object for extracting embeddings from donorA.
        - donorB_embedding_getter: A GraphModulePlus object for extracting embeddings from donorB.
    """

    # Set up models and data if not already initialized
    donorA.maybe_initialize()
    donorB.maybe_initialize()

    # Run dummy data through modelA up to layerA to get its shape
    donorA_embedding_getter = GraphModulePlus.new_from_copy(donorA.model).set_output(donorA.layer)
    dataA_shape = donorA.dataset.shape
    repA_shape = donorA_embedding_getter(torch.zeros((1,) + dataA_shape)).shape

    # Run dummy data through modelB up to layerB to get its shape
    donorB_embedding_getter = GraphModulePlus.new_from_copy(donorB.model).set_output(donorB.layer)
    dataB_shape = donorB.dataset.shape
    repB_shape = donorB_embedding_getter(torch.zeros((1,) + dataB_shape)).shape

    # Initialize the stitching layer
    stitching_layer = create_stitching_layer(repA_shape[1:], repB_shape[1:], stitch_family)

    # sanity-check that the stiching layer is the correct size
    assert stitching_layer(torch.zeros(repA_shape)).shape == repB_shape

    # retrieve next node for stitching the graph
    layerB_next = donorB.model.users_of(donorB.layer)
    assert len(layerB_next) == 1, "Stitching layers with |users| != 1 not supported"
    layerB_next = layerB_next[0].name

    # Stitch the stitching layer to the bottom of A_ subgraph
    modelAxB = GraphModulePlus.new_from_merge(
        {"donorA": donorA.model, "stitching_layer": stitching_layer, "donorB": donorB.model},
        {
            "stitching_layer": ["donorA_" + donorA.layer],
            "donorB_" + layerB_next: ["stitching_layer"],
        },
        auto_trace=False,
    )

    image = modelAxB.to_dot().create_png(prog="dot")
    with open(mlflow.get_artifact_uri("hybrid-model.png"), "wb") as f:
        f.write(image)

    return modelAxB, donorA_embedding_getter, donorB_embedding_getter


def loss_metrics_helper(
    target_type: TargetType,
    modelAxB: nn.Module,
    modelA: nn.Module,
    modelB: nn.Module,
    im: torch.Tensor,
    la: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to compute loss and metrics for the given target type."""

    out = modelAxB(im)
    task_acc = (torch.argmax(out, dim=-1) == la).float().mean()
    task_ce = nn.functional.cross_entropy(out, la)
    match target_type:
        case TargetType.TASK:
            loss = task_ce
        case TargetType.MATCH_UPSTREAM:
            teacher_prob = torch.softmax(modelA(im), dim=-1)
            loss = nn.functional.cross_entropy(out, teacher_prob)
        case TargetType.MATCH_DOWNSTREAM:
            teacher_prob = torch.softmax(modelB(im), dim=-1)
            loss = nn.functional.cross_entropy(out, teacher_prob)
        case _:
            assert_never(target_type)
    return loss, task_acc, task_ce


def _hash_tensor(tensor: torch.Tensor) -> float:
    """Returns a hash of the tensor, where, by 'hash' we just mean the floating point sum. We move
    values to CPU to make sure it is deterministic.
    """
    return tensor.detach().cpu().numpy().sum() if tensor is not None else 0.0


@torch.no_grad()
def snapshot_and_test(hashes, models, val_data, prefix, num_classes, device):
    """Snapshot params and run validation."""
    # Saving all three models so we can run sanity-checks later that only the parameters we
    # wanted to change actually changed in each phase.
    for name, model in models.items():
        # Only record the keys of the state dict that have changed from the donor model. This
        # is like what we would get if we had used GraphModulePlus.delta_state_dict(), but by doing
        # it with hashes we save on space. Load these snapshots using
        # GraphModulePlus.load_delta_state_dict() and the original donor model.
        sd_hash = hashes[name]
        delta_state_dict = {
            k: v for k, v in model.state_dict().items() if _hash_tensor(v) != sd_hash[k]
        }
        save_as_artifact(delta_state_dict, Path("weights") / f"{prefix}-{name}.pt")

    metrics = {
        "acc1": Accuracy("multiclass", num_classes=num_classes, top_k=1).to(device),
        "acc5": Accuracy("multiclass", num_classes=num_classes, top_k=5).to(device),
        "loss": nn.functional.cross_entropy,
    }
    values = defaultdict(lambda: torch.zeros(1, device=device))
    for im, la in tqdm(val_data, desc=f"Validating[{prefix}]", total=len(val_data)):
        im, la = im.to(device), la.to(device)
        for model_name, model in models.items():
            out = model(im)
            for metric_name, metric_fn in metrics.items():
                values[f"{prefix}-{model_name}-val-{metric_name}"] += metric_fn(out, la)

            # keeping track of cross entropy to donor models for sanity checking that the data is trending correctly
            teacher_prob = torch.softmax(models["modelA"](im), dim=-1)
            values[f"{prefix}-{model_name}-sanity-upstream_ce"] += nn.functional.cross_entropy(
                out, teacher_prob
            )
            teacher_prob = torch.softmax(models["modelB"](im), dim=-1)
            values[f"{prefix}-{model_name}-sanity-downstream_ce"] += nn.functional.cross_entropy(
                out, teacher_prob
            )

    # Average the values over the number of batches and log to mlflow
    values = {k: v.item() / len(val_data) for k, v in values.items()}
    mlflow.log_metrics(values)


def run_analysis(
    donorA: DonorSpec,
    donorB: DonorSpec,
    stitch_family: str,
    target_type: TargetType,
    init_batches: int,
    stitching_lr: float = 1e-3,
    downstream_lr: float = 1e-5,
    downstream_batches: int = 1000,
    batch_size: int = 200,
    num_workers: int = 4,
    device: torch.device | str = "cuda",
):

    # need to save name later, to fix deepcopy issue
    donorBname = donorB.model

    # Set up models and data if not already initialized
    donorA.maybe_initialize()
    donorB.maybe_initialize()

    modelAxB, embedding_getter_A, embedding_getter_B = create_hybrid_model(
        donorA, donorB, stitch_family
    )

    # Get a hash of each parameter of each model so that later we can ensure that we only snapshot
    # and store the parameters that *changed* during training. This makes the assumption that it
    # will be easy in the future to restore the original parameters from the donor models and load
    # in the delta_state_dicts (which is a feature of GraphModulePlus).
    model_hashes = {
        "modelA": {k: _hash_tensor(v) for k, v in donorA.model.state_dict().items()},
        "modelB": {k: _hash_tensor(v) for k, v in donorB.model.state_dict().items()},
        "modelAxB": {k: _hash_tensor(v) for k, v in modelAxB.state_dict().items()},
    }

    # Ensure all models are in eval mode and on device
    modelA = donorA.model.eval().to(device)
    modelB = donorB.model.eval().to(device)
    modelAxB = modelAxB.eval().to(device)

    # Get dataloaders
    donorB.dataset.setup("fit")
    train_data = donorB.dataset.train_dataloader(
        batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    donorB.dataset.setup("val")
    val_data = donorB.dataset.val_dataloader(
        batch_size=batch_size, num_workers=num_workers, drop_last=True
    )

    # Phase zero: regression-based initialization
    run_regression_init(
        modelAxB, embedding_getter_A, embedding_getter_B, train_data, init_batches, device
    )
    del embedding_getter_A, embedding_getter_B

    snapshot_and_test(
        model_hashes,
        {"modelA": modelA, "modelB": modelB, "modelAxB": modelAxB},
        val_data,
        "regression",
        donorB.dataset.num_classes,
        device,
    )

    # Phase 1: Train the stitching layer to convergence
    train_stitching_layer_to_convergence(
        modelAxB=modelAxB,
        modelA=modelA,
        modelB=modelB,
        train_data=train_data,
        target_type=target_type,
        lr=stitching_lr,
        max_steps=10000,
        device=device,
    )
    snapshot_and_test(
        model_hashes,
        {"modelA": modelA, "modelB": modelB, "modelAxB": modelAxB},
        val_data,
        "stitching",
        donorB.dataset.num_classes,
        device,
    )

    # Phase 2: Train the downstream model
    train_downstream_model(
        modelAxB=modelAxB,
        modelA=modelA,
        modelB=modelB,
        train_data=train_data,
        target_type=target_type,
        lr=downstream_lr,
        max_steps=downstream_batches,
        device=device,
        downstream_name=donorBname,
    )
    snapshot_and_test(
        model_hashes,
        {"modelA": modelA, "modelB": modelB, "modelAxB": modelAxB},
        val_data,
        "downstream",
        donorB.dataset.num_classes,
        device,
    )


def train_downstream_model(
    modelAxB: GraphModulePlus,
    modelA: GraphModulePlus,
    modelB: GraphModulePlus,
    train_data: DataLoader,
    target_type: TargetType,
    lr: float,
    max_steps: int,
    device: str | torch.device,
    downstream_name: str,
):
    modelA.eval()
    modelAxB.eval()
    modelB.train()

    if target_type == TargetType.MATCH_DOWNSTREAM:
        state_copy = {k: v.clone() for k, v in modelB.state_dict().items()}
        new_graph = torch.fx.Graph()
        output_node = new_graph.graph_copy(torch.fx.symbolic_trace(modelB).graph, {})

        # insert in output node, graph_copy does not copy those
        if output_node is not None:
            new_graph.output(output_node)

        # create new GraphModulePlus that will be deep copy of modelB to be the teacher for soft label training
        name = None
        class_name = "teacher_" + modelB.__class__.__name__ if name is None else name
        modelB_teacher = GraphModulePlus(root=modelB, graph=new_graph, class_name=class_name)

        # load state_dict into empty GraphModulePlus for a deep copy
        modelB_teacher.load_state_dict(state_copy)

        # apply sanity to check to insure that the copy matches the pretrained model
        sanity_check_model = _get_pretrained_model_by_name(downstream_name)
        sanity_check_model.to(device=device)
        modelB_teacher.to(device=device)

        teacher_params_before = {k: v.clone() for k, v in modelB_teacher.named_parameters()}
        sanity_params = {k: v.clone() for k, v in sanity_check_model.named_parameters()}

        for k, v in sanity_check_model.named_parameters():
            assert torch.allclose(sanity_params[k], teacher_params_before[k])

        # display compututation graph for debugging
        # display_model_graph(modelB_teacher)

        # currently deep copy does not work, so we will use the entirely new model pulled for MATCH_DOWNSTREAM training
        modelB_teacher = sanity_check_model
    else:
        modelB_teacher = None

    optimizer = torch.optim.Adam(modelB.parameters(), lr=lr)
    with frozen(modelA, modelAxB.stitching_layer):
        for step, (im, la) in enumerate(tqdm(train_data, total=max_steps, desc="Fine-tuning")):
            if step == max_steps:
                break

            im, la = im.to(device), la.to(device)

            loss, task_acc, task_ce = loss_metrics_helper(
                target_type, modelAxB, modelA, modelB_teacher, im, la
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric("downstream-modelAxB-train-loss", loss.detach(), step=step)
            mlflow.log_metric("downstream-modelAxB-train-ce", task_ce.detach(), step=step)
            mlflow.log_metric("downstream-modelAxB-train-acc", task_acc.detach(), step=step)

    # sanity_check that the tensor clone function worked to make a deepcopy
    if target_type == TargetType.MATCH_DOWNSTREAM:
        teacher_params_after = {k: v.clone() for k, v in modelB_teacher.named_parameters()}

        for k, v in modelB.named_parameters():
            assert torch.allclose(teacher_params_before[k], teacher_params_after[k])


def train_stitching_layer_to_convergence(
    modelAxB: GraphModulePlus,
    modelA: GraphModulePlus,
    modelB: GraphModulePlus,
    train_data: DataLoader,
    target_type: TargetType,
    max_steps: int,
    lr: float,
    lr_time_constant: float = 100.0,
    parameter_convergence_eps: float = 1e-4,
    device: str | torch.device = "cuda",
):
    modelA.eval()
    modelB.eval()
    modelAxB.stitching_layer.train()
    optimizer = torch.optim.Adam(modelAxB.stitching_layer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1 / (1 + step / lr_time_constant)
    )
    converged = False

    with frozen(modelA, modelB):
        step = 0
        last_params = {
            k: v.detach().clone() for k, v in modelAxB.stitching_layer.named_parameters()
        }
        while not converged:
            for im, la in train_data:
                im, la = im.to(device), la.to(device)

                loss, task_acc, task_ce = loss_metrics_helper(
                    target_type, modelAxB, modelA, modelB, im, la
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                new_params = {
                    k: v.detach().clone() for k, v in modelAxB.stitching_layer.named_parameters()
                }
                with torch.no_grad():
                    delta_params = torch.tensor(
                        [
                            (new_p - last_p).abs().max()
                            for new_p, last_p in zip(new_params.values(), last_params.values())
                        ]
                    ).max()
                    last_params = new_params

                if delta_params < parameter_convergence_eps:
                    converged = True

                if converged or step >= max_steps:
                    break

                if step % 100 == 0:
                    print(f"Step: {step}\tLoss: {loss.item()}\tDelta params: {delta_params.item()}")

                mlflow.log_metric("stitching-modelAxB-train-loss", loss.detach(), step=step)
                mlflow.log_metric("stitching-modelAxB-train-ce", task_ce.detach(), step=step)
                mlflow.log_metric("stitching-modelAxB-train-acc", task_acc.detach(), step=step)
                mlflow.log_metric("stitching-delta-params", delta_params.detach(), step=step)

                step += 1

    mlflow.log_metric("steps to train stitching layer", step)
    mlflow.log_metric("stitching layer converged", converged)


@torch.no_grad()
def run_regression_init(
    modelAxB: GraphModulePlus,
    embedding_getter_A: GraphModulePlus,
    embedding_getter_B: GraphModulePlus,
    train_data: DataLoader,
    init_batches: int,
    device: str | torch.device,
):
    for i, (im, la) in enumerate(tqdm(train_data, total=init_batches, desc="Regression init")):
        if i >= init_batches:
            break

        im = im.to(device)
        modelAxB.stitching_layer.init_by_regression(
            embedding_getter_A(im),
            embedding_getter_B(im),
            batched=True,
            final_batch=i == init_batches - 1,
        )


def _flatten_dict(d: dict, key_sep="_") -> dict:
    """Flattens a nested dictionary."""
    out = {}

    def flatten(x: dict, name: str = ""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + key_sep)
        else:
            if name[:-1] in out:
                raise ValueError(f"Duplicate key created during flattening: {name[:-1]}")
            out[name[:-1]] = x

    flatten(d)
    return out


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser(
        prog="learnable-stitching experiment",
        description="Create an amalgam of stiched models between two given pretrained models "
        "then apply downstream learning with respect to given dataset and set label type",
    )
    parser.add_function_arguments(run_analysis)
    args = parser.parse_args()
    print(args)

    experiment_name = "learnable-stitching-v0.2"
    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
    mlflow.set_experiment(experiment_name)

    # check if the experiment has already been run
    params = _flatten_dict(args.as_dict())

    prior_runs = search_runs_by_params(
        experiment_name=experiment_name,
        finished_only=True,
        params=params,
        ignore={"device": ..., "num_workers": ...},
    )
    if not prior_runs.empty:
        print("Experiment already run with these parameters. Exiting.")
        exit(0)

    # set up run name
    match params["target_type"]:
        case TargetType.TASK:
            task_name = "task"
        case TargetType.MATCH_UPSTREAM:
            task_name = "upstream"
        case TargetType.MATCH_DOWNSTREAM:
            task_name = "downstream"
        case _:
            assert_never(params["target_type"])

    run_name = (
        params["donorA_model"]
        + "_"
        + params["donorA_layer"]
        + "_"
        + params["donorA_dataset"]
        + "-X-"
        + params["donorB_model"]
        + "_"
        + params["donorB_layer"]
        + "_"
        + params["donorB_dataset"]
        + "-"
        + params["stitch_family"]
        + "_"
        + task_name
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_flatten_dict(args.as_dict()))
        run_analysis(**parser.instantiate_classes(args).as_dict())
