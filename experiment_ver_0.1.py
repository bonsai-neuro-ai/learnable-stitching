from pathlib import Path
from typing import NamedTuple

import mlflow
import nn_lib.models.fancy_layers as fl
import nn_lib.models.graph_module_plus as gmp
import torch
import torch.nn as nn
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import get_pretrained_model
from nn_lib.models.utils import conv2d_shape_inverse, frozen
from nn_lib.utils import save_as_artifact
from tqdm.auto import tqdm, trange


class ModelSpec(NamedTuple):
    model: str
    dataset: str


def return_accuracy(model, images, labels):
    with torch.no_grad():
        output = model(images)
    acc = torch.mean(torch.argmax(output, dim=1) == labels).item()
    return acc


def record_loss_metrics(modelA, modelB, loss, images, labels, name, step):
    # record loss
    mlflow.log_metric(name + "-loss-AxB", loss.item(), step=step)

    # record loss of donors
    outA = modelA(images)
    lossA = torch.nn.functional.cross_entropy(outA, labels)
    mlflow.log_metric(name + "-loss-A", lossA.item(), step=step)

    outB = modelB(images)
    lossB = torch.nn.functional.cross_entropy(outB, labels)
    mlflow.log_metric(name + "-loss-B", lossB.item(), step=step)

    # record stitching penalities
    mlflow.log_metric(name + "-penalty-A", loss.item() - lossA.item(), step=step)
    mlflow.log_metric(name + "-penalty-B", loss.item() - lossB.item(), step=step)


# A function that records the validation accuracy of modelA, ModelB, and their stiched modelAxB
# Currently there does not seem to be a validation set in the dataset in nn.lib, although imagenet does have a validation set
def record_val_accuarcy(data_module, modelA, modelB, modelAxB, name, step):
    data_module.setup("test")
    data_loader = data_module.test_dataloader()

    accA = 0
    accB = 0
    accAxB = 0

    # calculate the accuarcy for each batch of the evalaution data set
    # the return accurarcy function forces the gradients to be froze, so validation data should not be leaking into the training
    for images, labels in data_loader:
        accA += return_accuracy(modelA, images, labels)
        accB += return_accuracy(modelB, images, labels)
        accAxB += return_accuracy(modelAxB, images, labels)

    mlflow.log_metric(name + "-validation-A", accA, step=step)
    mlflow.log_metric(name + "-validation-B", accB, step=step)
    mlflow.log_metric(name + "-validation-AxB", accAxB, step=step)


# A function that records the test accuracy of modelA, ModelB, and their stiched modelAxB
def record_test_accuarcy(data_module, modelA, modelB, modelAxB, name, step):
    data_module.setup("test")
    data_loader = data_module.test_dataloader()

    accA = 0
    accB = 0
    accAxB = 0

    # calculate the accuarcy for each batch of the evalaution data set
    # the return accurarcy function forces the gradients to be froze, so validation data should not be leaking into the training
    for images, labels in data_loader:
        accA += return_accuracy(modelA, images, labels)
        accB += return_accuracy(modelB, images, labels)
        accAxB += return_accuracy(modelAxB, images, labels)

    mlflow.log_metric(name + "-test-A", accA, step=step)
    mlflow.log_metric(name + "-test-B", accB, step=step)
    mlflow.log_metric(name + "-test-AxB", accAxB, step=step)


def create_and_record_stitched_model(
    data_module,
    modelA,
    modelB,
    layerA,
    layerB,
    stitch_family,
    label_type,
    device,
    init_batch_num: int,
    batch_bound: int,
    epochs: int,
):
    # prepare and setup data
    data_module.prepare_data()

    # Squashing the batch norms, so they can be frozen
    modelA = modelA.squash_all_conv_batchnorm_pairs()
    modelB = modelB.squash_all_conv_batchnorm_pairs()

    # Set up models on the CUDA gpus and put them in eval mode
    modelA = modelA.to(device).eval()
    modelB = modelB.to(device).eval()

    # extract the neural network models in the subgraphs of modelA and modelB with respect to the layers
    modelA_ = modelA.extract_subgraph(inputs=["x"], output=layerA)
    modelB_ = modelB.extract_subgraph(inputs=["x"], output=layerB)
    modelA_.to(device)
    modelB_.to(device)

    # dummy input needed to get the shape of subgraphs to construct the stitching layer correctly
    dummy_input = torch.zeros(data_module.shape, device=device).unsqueeze(0)
    dummy_a = modelA_(dummy_input)
    dummy_b = modelB_(dummy_input)

    # retrieve next node for stitching the graph
    layerB_next = modelB._resolve_nodes(layerB)[0].users
    layerB_next = next(iter(layerB_next)).name

    # Construct the stiching layer, in the future this will be an inputed class
    # select the in_features and out_features to be the channel dimension of the convolution dummy inputs, probably not neccessary
    # nn.sequential requires all inputs to be nn.module, thus cannot use resize,
    mode = "bilinear"
    stitching_layer = nn.Sequential(
        fl.Interpolate2d(
            size=conv2d_shape_inverse(
                out_shape=dummy_b.shape[-2:], kernel_size=1, stride=1, dilation=1, padding=0
            ),
            mode=mode,
        ),
        (
            fl.RegressableConv2d(
                in_channels=dummy_a.shape[1], out_channels=dummy_b.shape[1], kernel_size=1
            )
        ),
    )

    stitching_layer.to(device)
    assert (
        stitching_layer(dummy_a).shape == dummy_b.shape
    )  # ensure the stiching layer is the correct size

    # Stitch the stitching layer to the bottom of A_ subgraph
    modelAxB = gmp.GraphModulePlus.new_from_merge(
        {"donorA": modelA, "sl": stitching_layer, "donorB": modelB},
        {"sl": ["donorA_" + layerA], "donorB_" + layerB_next: ["sl"]},
        auto_trace=False,
    )
    modelAxB.to(device).eval()

    # Record parameters before any training so that we can sanity-check that only the correct things
    # are changing. The .clone() is necessary so we get a snapshot of the parameters at this point in
    # time, rather than a reference to the parameters which will be updated later.
    paramsA = {k: v.clone() for k, v in modelA.named_parameters()}
    paramsB = {k: v.clone() for k, v in modelB.named_parameters()}
    paramsAxB = {k: v.clone() for k, v in modelAxB.named_parameters()}

    # prepare data to train over an epoch
    data_module.setup("fit")
    data_loader = data_module.train_dataloader()

    # initialize the stitchig layer using regression for a training speed up
    # initialize a couple of batches according to the init_batch
    with torch.no_grad():
        reg_input = []
        reg_output = []
        for i, (images, labels) in tqdm(
            enumerate(data_loader),
            total=init_batch_num,
            desc="collecting batches for regression init",
        ):
            if i >= init_batch_num:
                break

            images = images.to(device)

            reg_input.append(stitching_layer[0](modelA_(images)))
            reg_output.append(modelB_(images))

    # applying init by regression to speed up training time
    stitching_layer[1].init_by_regression(torch.cat(reg_input), torch.cat(reg_output))
    del reg_input, reg_output  # free up memory

    # track steps for mlflow logger
    step = 0

    # freeze context for training the stitching layer
    with frozen(modelA, modelB):

        for epoch in trange(epochs, desc="Stitching Layer Training over Epochs"):

            modelAxB.sl.train()
            optimizer = torch.optim.Adam(modelAxB.parameters(), lr=1e-6)

            # start optimizing
            for images, labels in tqdm(data_loader, desc="Training the Stitching Layer"):

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = modelAxB(images)
                # print(labels)

                if label_type == "soft":
                    labels = nn.Softmax(modelB(images)).detach()

                # This is task loss, but could be updated to be soft-labels to optimize match to model B
                loss = torch.nn.functional.cross_entropy(output, labels)

                # backprop and adjust
                loss.backward()
                optimizer.step()

                # return_accuracy(modelAxB, images, labels, "ModelAxB") #sanity check to see if AxB improves

                record_loss_metrics(modelA, modelB, loss, images, labels, "Stitching", step)
                step += 1

            # save modelAxB state each epoch, as well as the state of the optimizer
            info = {
                "state_dict": modelAxB.state_dict(),
                "opt": optimizer.state_dict(),
            }
            save_as_artifact(info, Path("weights") / "stitching-modelAxB.pt", run_id=None)

            record_val_accuarcy(data_module, modelA, modelB, modelAxB, name="Stitching", step=step)

    # Assert that no parameters changed *except* for stitched_model.stitching_layer
    for k, v in modelA.named_parameters():
        assert torch.allclose(v, paramsA[k])
    for k, v in modelB.named_parameters():
        assert torch.allclose(v, paramsB[k])
    for k, v in modelAxB.named_parameters():
        if k.startswith("sl"):
            assert not torch.allclose(v, paramsAxB[k])
        else:
            assert torch.allclose(v, paramsAxB[k])

    # freeze context for downstream learning
    modelAxB.sl.eval()
    with frozen(modelA, stitching_layer):
        for images, labels in tqdm(data_loader, desc="Training the Downstream Model"):

            modelAxB.donorB.train()
            optimizer = torch.optim.Adam(modelAxB.parameters(), lr=1e-8)

            for i, (images, labels) in tqdm(
                enumerate(data_loader), total=batch_bound, desc="Downstrearm Learning"
            ):
                if i == batch_bound:
                    break

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = modelAxB(images)
                # print(labels)

                if label_type == "soft":
                    labels = nn.Softmax(modelB(images)).detach()

                loss = torch.nn.functional.cross_entropy(output, labels)

                # backprop and adjust
                loss.backward()
                optimizer.step()

                # record loss
                record_loss_metrics(modelA, modelB, loss, images, labels, "Downstream", step)
                step += 1

            # save modelAxB state each epoch, as well as the state of the optimizer
            info = {
                "state_dict": modelAxB.state_dict(),
                "opt": optimizer.state_dict(),
            }
            save_as_artifact(info, Path("weights") / "DownsteamLearning-modelAxB.pt", run_id=None)

            record_val_accuarcy(data_module, modelA, modelB, modelAxB, name="Downstream", step=step)


def main(
    dataset: str,
    modelA: gmp.GraphModulePlus,
    modelB: gmp.GraphModulePlus,
    layerA: str,
    layerB: str,
    stitch_family: str = "1x1Conv",
    label_type: str = "class",
    epochs: int = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_batch_num = 5
    batch_bound = -100  # no bound on the number of batches, training will do a full epoch

    # todo: implement COCO segementation
    data_module = ImageNetDataModule(
        root_dir="/data/datasets", batch_size=100, num_workers=10
    )  # dataset is currently hardcoded to be imagenet

    create_and_record_stitched_model(
        data_module,
        modelA,
        modelB,
        layerA,
        layerB,
        stitch_family,
        label_type,
        device,
        init_batch_num,
        batch_bound,
        epochs,
    )


if __name__ == "__main__":
    import argparse

    # have to create an hardcoded parser so I can set the layers properly to the add blocks in graph_module_plus
    parser = argparse.ArgumentParser(
        prog="learnable-stitching experiment",
        description="Create an amalgam of stiched models between two given pretrained models "
        + "then apply downstream learning with respect to given dataset and set label type",
    )
    # TODO defaults and help messages (maybe jsonargparse?)
    parser.add_argument("--dataset")
    parser.add_argument("--modelA")
    parser.add_argument("--modelB")
    parser.add_argument("--stitch_family", default="1x1Conv", type=str)
    parser.add_argument("--label_type", default="class", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    args = parser.parse_args()

    # Models needed to be loaded prior to experiment so we can correctly extract the names of all
    # the add layers otherwise we may miss an add block or try to stitch to an add block that
    # does not exist
    str_modelA = args.modelA
    str_modelB = args.modelB

    # load pretrained models
    modelA = get_pretrained_model(args.modelA)
    modelB = get_pretrained_model(args.modelB)

    # turn models into the neccessary graphmoduleplus object
    modelA = gmp.GraphModulePlus.new_from_trace(modelA)
    modelB = gmp.GraphModulePlus.new_from_trace(modelB)

    # get split points between layers, could be much more efficient
    layersA = []
    layersB = []

    for node in modelA.graph.nodes:  # All the stitch points for modelA
        if node.name.count("add") > 0:
            layersA.append(node.name)

    for node in modelB.graph.nodes:  # all the stitch points for modelB
        if node.name.count("add") > 0:
            layersB.append(node.name)

    # run an experiment for each combination of layers for each of the models
    for layerA in layersA:
        for layerB in layersB:
            mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
            mlflow.set_experiment("learnable--stitching")

            with mlflow.start_run(
                run_name=f"{args.dataset}--{args.modelA}_{layerA}--x{args.stitch_family}x--{args.modelB}_{layerB}--label_{args.label_type}"
            ):
                mlflow.log_params(
                    {
                        "layerA": layerA,
                        "layerB": layerB,
                        **vars(args),
                    }
                )
                main(
                    dataset=args.dataset,
                    modelA=modelA,
                    modelB=modelB,
                    layerA=layerA,
                    layerB=layerB,
                    stitch_family=args.stitch_family,
                    label_type=args.label_type,
                    epochs=args.epochs,
                )
