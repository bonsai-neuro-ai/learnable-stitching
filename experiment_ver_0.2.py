import dataclasses
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from typing import Self, assert_never

import mlflow
import torch
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
    with open("/tmp/hybrid-model.png", "wb") as f:
        f.write(image)
    mlflow.log_artifact("/tmp/hybrid-model.png", "hybrid-model.png")

    return modelAxB, donorA_embedding_getter, donorB_embedding_getter


def loss_metrics_helper(target_type, modelAxB, modelA, modelB, im, la):
    """Helper function to compute loss and metrics for the given target type.
    """
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


@torch.no_grad()
def snapshot_and_test(models, val_data, prefix, num_classes, device):
    """Snapshot params and run validation."""
    # Saving all three models so we can run sanity-checks later that only the parameters we
    # wanted to change actually changed in each phase.
    for name, model in models.items():
        save_as_artifact(model.state_dict(), Path("weights") / f"{prefix}-{name}.pt")

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
    # Set up models and data if not already initialized
    donorA.maybe_initialize()
    donorB.maybe_initialize()
    modelAxB, embedding_getter_A, embedding_getter_B = create_hybrid_model(
        donorA, donorB, stitch_family
    )

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
    )
    snapshot_and_test(
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
):
    modelA.eval()
    modelAxB.eval()
    modelB.train()

    if target_type == TargetType.MATCH_DOWNSTREAM:
        modelB_teacher = deepcopy(modelB)
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
    optimizer = torch.optim.Adam(modelAxB.parameters(), lr=lr)
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
    reg_input = []
    reg_output = []
    for i, (im, la) in enumerate(tqdm(train_data, total=init_batches, desc="Regression init")):
        if i >= init_batches:
            break

        im = im.to(device)
        reg_input.append(embedding_getter_A(im))
        reg_output.append(embedding_getter_B(im))

    # applying init by regression to speed up training time
    modelAxB.stitching_layer.init_by_regression(
        from_data=torch.cat(reg_input, dim=0).detach(),
        to_data=torch.cat(reg_output, dim=0).detach(),
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

    experiment_name = "learnable-stitching-v0.2"
    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
    mlflow.set_experiment(experiment_name)

    # check if the experiment has already been run
    prior_runs = search_runs_by_params(
        experiment_name=experiment_name,
        finished_only=True,
        params=_flatten_dict(args.as_dict()),
        skip_fields={"device": ..., "num_workers": ...},
    )
    if not prior_runs.empty:
        print("Experiment already run with these parameters. Exiting.")
        exit(0)

    with mlflow.start_run():
        mlflow.log_params(_flatten_dict(args.as_dict()))
        run_analysis(**parser.instantiate_classes(args).as_dict())
