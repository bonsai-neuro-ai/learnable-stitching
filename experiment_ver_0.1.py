import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Self

import mlflow
import torch
import torch.nn as nn
from graphene.utils.dataloader import DataLoader
from nn_lib.datasets import ImageNetDataModule, TorchvisionDataModuleBase
from nn_lib.models import get_pretrained_model
from nn_lib.models.graph_module_plus import GraphModulePlus
from nn_lib.models.utils import frozen
from nn_lib.utils import save_as_artifact
from torchmetrics import Accuracy
from tqdm.auto import tqdm

from stitching import create_stitching_layer


def _get_dataset_by_name(name: str) -> TorchvisionDataModuleBase:
    # todo: implement other datasets
    if name == "imagenet":
        dm = ImageNetDataModule(root_dir="/data/datasets")
    else:
        raise ValueError("Only ImageNet supported so far")
    dm.prepare_data()
    return dm


def _get_pretrained_model_by_name(name: str) -> GraphModulePlus:
    torchvision_model = get_pretrained_model(name)
    return GraphModulePlus.new_from_trace(torchvision_model).squash_all_conv_batchnorm_pairs()


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

    return modelAxB, donorA_embedding_getter, donorB_embedding_getter


def run_analysis(
    donorA: DonorSpec,
    donorB: DonorSpec,
    stitch_family: str,
    label_type: str,
    init_batches: int,
    downstream_batches: int,
    batch_size: int,
    num_workers: int,
    device: torch.device | str,
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
    donorA.dataset.setup("fit")
    train_data = donorA.dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)

    donorA.dataset.setup("val")
    val_data = donorA.dataset.val_dataloader(batch_size=batch_size, num_workers=num_workers)

    @torch.no_grad()
    def snapshot_and_validate(prefix: str):
        """Snapshot params and run validation."""
        nonlocal modelA, modelB, modelAxB, val_data
        # Saving all three models so we can run sanity-checks later that only the parameters we
        # wanted to change actually changed in each phase.
        save_as_artifact(modelAxB.state_dict(), Path("weights") / f"{prefix}-modelAxB.pt")
        save_as_artifact(modelA.state_dict(), Path("weights") / f"{prefix}-modelA.pt")
        save_as_artifact(modelB.state_dict(), Path("weights") / f"{prefix}-modelB.pt")

        models = {"modelA": modelA, "modelB": modelB, "modelAxB": modelAxB}
        metrics = {
            "acc1": Accuracy("multiclass", num_classes=donorB.dataset.num_classes, top_k=1).to(
                device
            ),
            "acc5": Accuracy("multiclass", num_classes=donorB.dataset.num_classes, top_k=5).to(
                device
            ),
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

    # Snapshot everything before doing any training
    snapshot_and_validate("init")

    # Phase zero: regression-based initialization
    run_regression_init(
        modelAxB, embedding_getter_A, embedding_getter_B, train_data, init_batches, device
    )
    snapshot_and_validate("regression")

    # Phase 1: Train the stitching layer to convergence
    max_steps_part_1 = int(2 * len(train_data.dataset) / batch_size)
    train_stitching_layer_to_convergence(
        modelAxB=modelAxB,
        modelA=modelA,
        modelB=modelB,
        train_data=train_data,
        label_type=label_type,
        device=device,
        max_steps=max_steps_part_1,
    )
    snapshot_and_validate("stitching")

    # Phase 2: Train the downstream model
    train_downstream_model(
        modelAxB=modelAxB,
        modelA=modelA,
        modelB=modelB,
        train_data=train_data,
        label_type=label_type,
        device=device,
        max_steps=downstream_batches,
        init_step=max_steps_part_1 + 1,
    )
    snapshot_and_validate("downstream")


def train_downstream_model(
    modelAxB: GraphModulePlus,
    modelA: GraphModulePlus,
    modelB: GraphModulePlus,
    train_data: DataLoader,
    label_type: str,
    device: str | torch.device,
    max_steps: int,
    init_step: int,
):
    modelA.eval()
    modelAxB.eval()
    modelB.train()
    optimizer = torch.optim.Adam(modelB.parameters(), lr=1e-8)
    with frozen(modelA, modelAxB.stitching_layer):
        for i, (im, la) in enumerate(tqdm(train_data, total=max_steps, desc="Fine-tuning")):
            if i == max_steps:
                break

            optimizer.zero_grad()
            im, la = im.to(device), la.to(device)
            output = modelAxB(im)
            target = _soft_target_helper(im, la, modelA, modelB, label_type)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            mlflow.log_metric("train_loss", loss.detach(), step=init_step + i)


def train_stitching_layer_to_convergence(
    modelAxB: GraphModulePlus,
    modelA: GraphModulePlus,
    modelB: GraphModulePlus,
    train_data: DataLoader,
    label_type: str,
    device: str | torch.device,
    max_steps: int,
    parameter_convergence_eps: float = 1e-4,
):
    modelA.eval()
    modelB.eval()
    modelAxB.stitching_layer.train()
    optimizer = torch.optim.Adam(modelAxB.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    converged = False

    with frozen(modelA, modelB):
        step = 0
        last_params = {
            k: v.detach().clone() for k, v in modelAxB.stitching_layer.state_dict().items()
        }
        while not converged:
            for im, la in train_data:
                optimizer.zero_grad()
                im, la = im.to(device), la.to(device)
                output = modelAxB(im)
                target = _soft_target_helper(im, la, modelA, modelB, label_type)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()

                new_params = {
                    k: v.detach().clone() for k, v in modelAxB.stitching_layer.state_dict().items()
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

                mlflow.log_metric("train_loss", loss.detach(), step=step)
                mlflow.log_metric("delta_params", delta_params.detach(), step=step)

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
        from_data=torch.cat(reg_input, dim=0),
        to_data=torch.cat(reg_output, dim=0),
    )


def _soft_target_helper(images, labels, modelA, modelB, loss_type: str):
    if loss_type == "class":
        return labels
    elif loss_type == "soft-A":
        with torch.no_grad():
            return torch.softmax(modelA(images), dim=-1)
    elif loss_type == "soft-B":
        raise NotImplementedError(
            "Soft targets for modelB not implemented. Tricky because we need to be careful if B"
            " is changing while we train AxB!"
        )
    else:
        raise ValueError("Invalid loss type")


if __name__ == "__main__":
    import argparse

    # have to create an hardcoded parser so I can set the layers properly to the add blocks in graph_module_plus
    parser = argparse.ArgumentParser(
        prog="learnable-stitching experiment",
        description="Create an amalgam of stiched models between two given pretrained models "
        + "then apply downstream learning with respect to given dataset and set label type",
    )
    # TODO defaults and help messages (maybe jsonargparse?)
    parser.add_argument("--modelA")
    parser.add_argument("--modelB")
    parser.add_argument("--datasetA", default="imagenet", type=str)
    parser.add_argument("--datasetB", default="imagenet", type=str)
    parser.add_argument("--stitch_family", default="1x1Conv", type=str)
    parser.add_argument("--label_type", default="class", type=str)
    parser.add_argument("--init_batches", default=10, type=int)
    parser.add_argument("--downstream_batches", default=100, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--device", default="cuda", type=torch.device)
    args = parser.parse_args()

    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
    mlflow.set_experiment("learnable--stitching")

    tmpA = _get_pretrained_model_by_name(args.modelA)
    tmpB = _get_pretrained_model_by_name(args.modelA)

    layersA = [l.name for l in tmpA.graph.nodes if l.name.count("add") > 0]
    layersB = [l.name for l in tmpB.graph.nodes if l.name.count("add") > 0]

    # run an experiment for each combination of layers for each of the models
    for layerA in layersA:
        for layerB in layersB:
            with mlflow.start_run(
                run_name=f"{args.modelA}_{layerA}--x{args.stitch_family}x--{args.modelB}_{layerB}--label_{args.label_type}"
            ):
                mlflow.log_params(
                    {
                        "layerA": layerA,
                        "layerB": layerB,
                        **vars(args),
                    }
                )

                run_analysis(
                    donorA=DonorSpec(args.modelA, layerA, args.datasetA),
                    donorB=DonorSpec(args.modelB, layerB, args.datasetB),
                    stitch_family=args.stitch_family,
                    label_type=args.label_type,
                    init_batches=args.init_batches,
                    downstream_batches=args.downstream_batches,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                )
