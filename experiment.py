import dataclasses
import itertools
import sys
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum, auto
from pathlib import Path
from typing import Literal, Self, assert_never

import mlflow
import torch
import torch.fx.graph
from graphene.utils.dataloader import DataLoader
from nn_lib.datasets import ImageNetDataModule, TorchvisionDataModuleBase
from nn_lib.models import get_pretrained_model
from nn_lib.models.graph_module_plus import GraphModulePlus
from nn_lib.optim import LRFinder
from nn_lib.optim.lr_finder import UnstableLREstimate
from nn_lib.utils import RunIndex, logged_run, save_as_artifact
from nn_lib.utils.models import frozen
from torch import nn
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
            raise ValueError(f"Unknown dataset {dataset_name}")


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
        donorA: DonorSpec, donorB: DonorSpec, stitch_family: str, plot_architecture: bool = False
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

    if plot_architecture:
        image = donorA.model.to_dot().create_png(prog="dot")
        with open("donorA-computation-map.png", "wb") as f:
            f.write(image)

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
        outAxB: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to compute loss and metrics for the given target type."""
    if outAxB is None:
        outAxB = modelAxB(im)
    task_acc = (torch.argmax(outAxB, dim=-1) == la).float().mean()
    task_ce = nn.functional.cross_entropy(outAxB, la)
    match target_type:
        case TargetType.TASK:
            loss = task_ce
        case TargetType.MATCH_UPSTREAM:
            teacher_prob = torch.softmax(modelA(im), dim=-1)
            loss = nn.functional.cross_entropy(outAxB, teacher_prob)
        case TargetType.MATCH_DOWNSTREAM:
            teacher_prob = torch.softmax(modelB(im), dim=-1)
            loss = nn.functional.cross_entropy(outAxB, teacher_prob)
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

    # the metrics listed are measured on the validation set of the data
    metrics = {
        "acc1": Accuracy("multiclass", num_classes=num_classes, top_k=1).to(
            device, non_blocking=True
        ),
        "acc5": Accuracy("multiclass", num_classes=num_classes, top_k=5).to(
            device, non_blocking=True
        ),
        "loss": nn.functional.cross_entropy,
    }
    values = defaultdict(lambda: torch.zeros(1, device=device))
    for im, la in tqdm(val_data, desc=f"Validating[{prefix}]", total=len(val_data)):
        im, la = im.to(device, non_blocking=True), la.to(device, non_blocking=True)
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

    # Average the validation metric values over the number of batches and log to mlflow
    values = {k: v.item() / len(val_data) for k, v in values.items()}
    mlflow.log_metrics(values, synchronous=False)


@dataclasses.dataclass
class TrainingConfig:
    target_type: TargetType = TargetType.TASK
    optimizer: Literal["sgd", "adam"] = "adam"
    lr: float | Literal["auto"] = "auto"
    val_patience: int = 3
    val_every: int = 20
    quick_val_batches: int = 10
    max_steps: int = 10000
    use_best_val: bool = True
    smallest_reasonable_lr: float = 1e-8
    largest_reasonable_lr: float = 0.1


_DEFAULT_TRAINING_CONFIG = TrainingConfig()


def training_helper(
        modelAxB: GraphModulePlus,
        modelA: GraphModulePlus,
        modelB: GraphModulePlus,
        train_data: DataLoader,
        val_data: DataLoader,
        frozen_modules: list[nn.Module],
        parameters: Iterable[nn.Parameter],
        logging_prefix: str,
        config: TrainingConfig,
        device: str | torch.device,
):
    # find the largest learning rate that does not cause the model to jump out of the loss incline
    if config.lr == "auto":

        def criterion(inputs, outputs, targets):
            return loss_metrics_helper(
                config.target_type, modelAxB, modelA, modelB, inputs, targets, outAxB=outputs
            )[0]

        parameters, tmp_parameters = itertools.tee(parameters)
        if config.optimizer == "sgd":
            optim = torch.optim.SGD(tmp_parameters, momentum=0.9, lr=1e-6)
        elif config.optimizer == "adam":
            optim = torch.optim.Adam(tmp_parameters, lr=1e-6)
        else:
            raise ValueError(
                f"Learning rate finder only works with SGD or Adam optimizers, not {config.optimizer}"
            )

        lr_finder = LRFinder(modelAxB, optim, criterion=criterion, device=device)
        lr_finder.range_test(
            train_data,
            start_lr=config.smallest_reasonable_lr / 10.0,
            end_lr=config.largest_reasonable_lr * 10.0,
            num_iter=100,
            callback=lambda itr, lr, loss: mlflow.log_metrics(
                {
                    logging_prefix + "-lr-finder-lr": lr,
                    logging_prefix + "-lr-finder-loss": loss,
                },
                step=itr,
                synchronous=False,
            ),
        )
        try:
            init_lr = min(
                config.largest_reasonable_lr,
                max(config.smallest_reasonable_lr, lr_finder.suggestion()),
            )
        except UnstableLREstimate:
            init_lr = config.smallest_reasonable_lr
    else:
        init_lr = config.lr

    mlflow.log_metric(logging_prefix + "-init-lr", init_lr)

    converged_lr = max(init_lr / 1000, config.smallest_reasonable_lr / 2)

    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=init_lr, momentum=0.9)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=init_lr)
    else:
        raise ValueError(f"Unknown optimizer option {config.optimizer}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.val_patience,
        # avoid floating point issues with <= comparisons
        min_lr=converged_lr / 1.0000000001,
        # we do our own LR < min_LR check; disable the 'eps' in the scheduler to avoid it
        # refusing to make the LR small (which then causes us to never 'converge')
        eps=0.0,
    )
    val_loss_key = logging_prefix + "-modelAxB-quick-val-loss"

    @torch.no_grad()
    def quick_validation_loss(
            val_data_: DataLoader, max_batches: int | None = None,
            device_: str | torch.device = "cuda"
    ) -> dict[str, torch.Tensor]:
        num_batches = len(val_data_) if max_batches is None else min(len(val_data_), max_batches)
        metrics = defaultdict(lambda: torch.zeros(1, device=device_))
        for i, (im, la) in enumerate(val_data_):
            im, la = im.to(device_, non_blocking=True), la.to(device_, non_blocking=True)
            loss, task_acc, task_ce = loss_metrics_helper(
                config.target_type, modelAxB, modelA, modelB, im, la
            )
            metrics[logging_prefix + "-modelAxB-quick-val-loss"] += loss / num_batches
            metrics[logging_prefix + "-modelAxB-quick-val-acc"] += task_acc / num_batches
            metrics[logging_prefix + "-modelAxB-quick-val-ce"] += task_ce / num_batches

            if i + 1 >= num_batches:
                break
        return metrics

    best_val, converged, errored = float("inf"), False, False
    best_snapshot = {
        k: v.detach().clone() for k, v in modelAxB.state_dict().items()
    }

    with frozen(*frozen_modules):
        step = 0

        # The stitching layer must be trained to convergence to prevent the downstream learning
        # from *picking up the slack* of the stitching layers unconverged training which would
        # CONFOUND THE RESULTS!
        while not (converged or errored or step >= config.max_steps):
            for im, la in tqdm(
                    train_data,
                    desc=f"Training[{logging_prefix}]",
                    initial=step,
                    total=config.max_steps,
                    leave=False,
            ):
                im, la = im.to(device, non_blocking=True), la.to(device, non_blocking=True)

                optimizer.zero_grad()
                loss, task_acc, task_ce = loss_metrics_helper(
                    config.target_type, modelAxB, modelA, modelB, im, la
                )
                loss.backward()
                optimizer.step()

                if step % config.val_every == 0:
                    val_metrics = quick_validation_loss(val_data, config.quick_val_batches, device)
                    scheduler.step(val_metrics[val_loss_key])
                    converged = all(group["lr"] <= converged_lr for group in optimizer.param_groups)

                    mlflow.log_metrics(val_metrics, step=step, synchronous=False)

                    if val_metrics[val_loss_key] < best_val:
                        best_val = val_metrics[val_loss_key]
                        best_snapshot = {
                            k: v.detach().clone() for k, v in modelAxB.state_dict().items()
                        }

                    if not all(map(torch.isfinite, val_metrics.values())):
                        errored = True
                        break

                    mlflow.log_metric(
                        logging_prefix + "-lr", optimizer.param_groups[0]["lr"], step=step
                    )

                mlflow.log_metrics(
                    {
                        logging_prefix + "-modelAxB-train-loss": loss.detach(),
                        logging_prefix + "-modelAxB-train-ce": task_ce.detach(),
                        logging_prefix + "-modelAxB-train-acc": task_acc.detach(),
                    },
                    step=step,
                    synchronous=False,
                )

                if converged or step >= config.max_steps:
                    break

                step += 1

    if config.use_best_val or errored:
        modelAxB.load_state_dict(best_snapshot)

    mlflow.log_metric(logging_prefix + " total steps", step)
    mlflow.log_metric(logging_prefix + " converged", converged)
    mlflow.log_metric(logging_prefix + " errored", errored)


def run_analysis(
        donorA: DonorSpec,
        donorB: DonorSpec,
        stitch_family: str,
        finetune_init: Literal[
            "random", "regression", "regression+stitch", "random+stitch"
        ] = "regression+stitch",
        finetune_stitching_layer: bool = False,
        init_batches: int = 1000,
        stitching_config: TrainingConfig = _DEFAULT_TRAINING_CONFIG,
        finetune_config: TrainingConfig = _DEFAULT_TRAINING_CONFIG,
        batch_size: int = 200,
        num_workers: int = 4,
        device: str | torch.device = "cuda",
):
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
    if finetune_init in ["regression", "regression+stitch"]:
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

    ############################################
    # Phase 1: Train the stitching layer alone #
    ############################################
    modelA.eval()
    modelAxB.eval()
    modelB.eval()
    modelAxB.stitching_layer.train()

    training_helper(
        modelAxB=modelAxB,
        modelA=modelA,
        modelB=modelB,
        train_data=train_data,
        val_data=val_data,
        frozen_modules=[modelA, modelB],
        parameters=modelAxB.stitching_layer.parameters(),
        logging_prefix="stitching",
        config=stitching_config,
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

    #######################################
    # Phase 2: Train the downstream model #
    #######################################
    modelA.eval()
    modelAxB.eval()
    modelB.train()

    if finetune_stitching_layer:
        modelAxB.stitching_layer.train()
        trainable_parameters = itertools.chain(
            modelAxB.stitching_layer.parameters(), modelB.parameters()
        )
    else:
        trainable_parameters = modelB.parameters()

    training_helper(
        modelAxB=modelAxB,
        modelA=modelA,
        modelB=modelB,
        train_data=train_data,
        val_data=val_data,
        frozen_modules=[modelA],
        parameters=trainable_parameters,
        logging_prefix="finetuning",
        config=finetune_config,
        device=device,
    )

    snapshot_and_test(
        model_hashes,
        {"modelA": modelA, "modelB": modelB, "modelAxB": modelAxB},
        val_data,
        "finetuning",
        donorB.dataset.num_classes,
        device,
    )


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

        im = im.to(device, non_blocking=True)
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
            for key, val in x.items():
                flatten(val, name + key + key_sep)
        else:
            if name[:-1] in out:
                raise ValueError(f"Duplicate key created during flattening: {name[:-1]}")
            out[name[:-1]] = x

    flatten(d)
    return out


if __name__ == "__main__":
    import os

    import jsonargparse
    from dotenv import load_dotenv

    load_dotenv()  # loads .env file into environment variables

    parser = jsonargparse.ArgumentParser(
        prog="learnable-stitching experiment",
        description="Create an amalgam of stiched models between two given pretrained models "
                    "then apply downstream learning with respect to given dataset and set label type",
    )
    parser.add_function_arguments(run_analysis)
    parser.add_argument("--mlflow_experiment", type=str, default="debug")
    parser.add_argument("--config", action="config")
    args = parser.parse_args()

    # The --config argument has done its job at this point, so we remove it from the args
    # namespace to avoid passing it to run_analysis.
    args.pop("config")

    user = os.getenv("MLFLOW_SQL_USERNAME")
    password = os.getenv("MLFLOW_SQL_PASSWORD")
    db_name = os.getenv("MLFLOW_SQL_DATABASE")

    mlflow.set_tracking_uri(f"mysql+pymysql://{user}:{password}@localhost:3306/{db_name}")
    experiment_name = args.pop("mlflow_experiment")
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            experiment_name, artifact_location=f"/projects/learnable-stitching/{experiment_name}/"
        )
    mlflow.set_experiment(experiment_name)

    index = RunIndex.from_experiment(args, experiment_names=[experiment_name])
    if args in index:
        print(f"Run with parameters {args} already exists in experiment {experiment_name}.")
        sys.exit(0)

    # set up run name
    params = _flatten_dict(args.as_dict())

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
            + str(params["stitching_config_target_type"]).split(".")[-1].lower()
            + "_"
            + params["finetune_init"]
    )

    # set up mlflow to record the run, log parameter experiments, and start the process to generate data for learnable stitching experiment
    with logged_run(args, index):
        run_analysis(**parser.instantiate(args).as_dict())
