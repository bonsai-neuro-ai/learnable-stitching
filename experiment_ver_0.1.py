#path libraries
from pathlib import Path
from typing import NamedTuple

#Neural network and machine learning libraries
import torch
import torch.nn as nn
import nn_lib.models.graph_module_plus as gmp
import nn_lib.models.fancy_layers as fl
from nn_lib.models.utils import conv2d_shape_inverse, frozen
from nn_lib.datasets import ImageNetDataModule, CocoDetectionDataModule
from nn_lib.models import get_pretrained_model, graph_utils

#logging libraries
import mlflow
from nn_lib.utils import save_as_artifact

from tqdm.auto import tqdm


class ModelSpec(NamedTuple):
    model: str
    dataset: str

def return_accuracy(model, images, labels, name):
    with torch.no_grad():
        output = model(images)
    acc = torch.sum(torch.argmax(output, dim=1) == labels).item() / len(labels)
    #print(f"{name}: {torch.sum(torch.argmax(output, dim=1) == labels).item()} / {len(labels)}") #print statement for sanity checks
    return acc

def create_and_test_stitched_model(data_module, modelA, modelB, layerA, layerB, stitch_family, label_type, device,  init_batch_num :int, batch_bound :int, epochs :int):
    #prepare and setup data
    data_module.prepare_data()

    #turn models into the neccessary graphmoduleplus object
    modelA = gmp.GraphModulePlus.new_from_trace(modelA)
    modelB = gmp.GraphModulePlus.new_from_trace(modelB)

    #Squashing the batch norms, so they can be frozen
    modelA = modelA.squash_all_conv_batchnorm_pairs()
    modelB = modelB.squash_all_conv_batchnorm_pairs()

    # Set up models on the CUDA gpus and put them in eval mode
    modelA = modelA.to(device).eval()
    modelB = modelB.to(device).eval()

    #extract the neural network models in the subgraphs of modelA and modelB with respect to the layers
    modelA_ = modelA.extract_subgraph(inputs = ["x"], output=layerA)
    modelB_ = modelB.extract_subgraph(inputs = ["x"], output=layerB)
    modelA_.to(device)
    modelB_.to(device)
    
    #dummy input needed to get the shape of subgraphs to construct the stitching layer correctly
    dummy_input = torch.zeros(data_module.shape, device=device).unsqueeze(0)
    dummy_a = modelA_(dummy_input)
    dummy_b = modelB_(dummy_input)

    #retrieve next node for stitching the graph
    layerB_next = modelB._resolve_nodes(layerB)[0].users
    layerB_next = next(iter(layerB_next)).name
    
    #Construct the stiching layer, in the future this will be an inputed class
    #select the in_features and out_features to be the channel dimension of the convolution dummy inputs, probably not neccessary
    #nn.sequential requires all inputs to be nn.module, thus cannot use resize, 
    mode = "bilinear"
    stitching_layer = nn.Sequential(
        fl.Interpolate2d(size=conv2d_shape_inverse(out_shape=dummy_b.shape[-2:], kernel_size=1, stride = 1, dilation = 1, padding = 0), mode = mode),
        (fl.RegressableConv2d(in_channels=dummy_a.shape[1], out_channels=dummy_b.shape[1], kernel_size=1))
    )

    stitching_layer.to(device)
    assert stitching_layer(dummy_a).shape == dummy_b.shape #ensure the stiching layer is the correct size

    #Stitch the stitching layer to the bottom of A_ subgraph
    modelAxB = modelA_.new_from_merge({ "donorA": modelA, "sl": stitching_layer, "donorB": modelB}, 
                                    {"sl": ["donorA_"+layerA], "donorB_"+layerB_next: ["sl"]}, 
                                    auto_trace = False)
    modelAxB.to(device).eval()
    
    #Record parameters before any training so that we can sanity-check that only the correct things
    # are changing. The .clone() is necessary so we get a snapshot of the parameters at this point in
    # time, rather than a reference to the parameters which will be updated later.
    paramsA = {k: v.clone() for k, v in modelA.named_parameters()}
    paramsB = {k: v.clone() for k, v in modelB.named_parameters()}
    paramsAxB = {k: v.clone() for k, v in modelAxB.named_parameters()}

    #prepare data to train over an epoch
    data_module.setup("fit")
    data_loader = data_module.train_dataloader()


    #initialize the stitchig layer using regression for a training speed up
    #initialize a couple of batches according to the init_batch 
    reg_input = []
    reg_output = []
    for i, (images, labels) in tqdm(enumerate(data_loader), total= init_batch_num, desc = "collecting batches for regression init"):
        if (i >= init_batch_num): 
            break

        images = images.to(device)
        labels = labels.to(device)

        reg_input.append(stitching_layer[0](modelA_(images)))
        reg_output.append(modelB_(images))

        return_accuracy(modelA, images, labels, "ModelA")
        return_accuracy(modelB, images, labels, "ModelB")
        return_accuracy(modelAxB, images, labels, "ModelAxB")
    
    #applying init by regression to speed up training time
    stitching_layer[1].init_by_regression(torch.cat(reg_input), torch.cat(reg_output))
    
    #freeze context for training the stitching layer
    with frozen(modelA, modelB):
        for epoch in range(epochs):
                
            modelAxB.sl.train()
            optimizer = torch.optim.Adam(modelAxB.parameters(), lr=0.000001)
            

            #needed if we want to downstream learn on the same data as the stiching layer was trained on, does not matter when completeting epochs
            images_2 = []
            labels_2 = []

            #start optimizing
            for i, (images, labels) in tqdm(enumerate(data_loader), total= batch_bound, desc="Training the Stitching Layer"):
                    if (i == batch_bound): 
                        break

                    images_2.append(images)
                    labels_2.append(labels)

                    images = images.to(device)
                    labels = labels.to(device)
            
                    optimizer.zero_grad()
                    output = modelAxB(images)
                    #print(labels)


                    if label_type == "soft": #soft labels
                        labels = nn.Softmax(modelA(images))

                        
                    # This is task loss, but could be updated to be soft-labels to optimize match to model B
                    loss = torch.nn.functional.cross_entropy(output, labels)

                    #backprop and adjust
                    loss.backward()
                    optimizer.step()

                    #return_accuracy(modelAxB, images, labels, "ModelAxB") #sanity check to see if AxB improves 
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


    #freeze context for downstream learning
    modelAxB.sl.eval()
    with frozen(modelA, stitching_layer):
        modelAxB.donorB.train()

        optimizer = torch.optim.Adam(modelAxB.parameters(), lr=0.000000001)
                        
        for i, (images, labels) in tqdm(enumerate(data_loader), total=batch_bound, desc= "Downstream Learning"):
        #for i, (images, labels) in tqdm(enumerate(zip(images_2,labels_2)), total=len(images_2), desc= "Downstream Learning"): #same images and labels used for the stitching layer
            if (i == batch_bound): 
                break

            images = images.to(device)
            labels = labels.to(device)
                    
            optimizer.zero_grad()
            output = modelAxB(images)
            #print(labels)


            #if label_type == "soft": #soft labels can be derrived by picking the largest value from the softmax of the logits
                #labels = nn.Softmax(modelA(images))

                                
            # This is task loss, but could be updated to be soft-labels to optimize match to model B
            loss = torch.nn.functional.cross_entropy(output, labels)

            #backprop and adjust
            loss.backward()
            optimizer.step()       

            #return_accuracy(modelB, images, labels, "ModelB") #sanity check to see if AxB improves 
            #return_accuracy(modelAxB, images, labels, "ModelAxB") #sanity check to see if AxB improves 
    
def main(dataset: str, modelA: str, modelB: str, layerA: str, layerB: str, stitch_family :str, label_type: str, epochs: int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_batch_num = 5
    batch_bound = 100 #no bound on the number of batches, training will do a full epoch

    data_module = ImageNetDataModule(root_dir="/data/datasets", batch_size = 100, num_workers = 10)
    modelA = get_pretrained_model("resnet18")
    modelB = get_pretrained_model("resnet34")
    layerA = "add_2"
    layerB = "add_5"

    stitch_family = "1x1Conv" #currently will not do anything 
    label_type = "class"

    epochs = 1

    create_and_test_stitched_model(data_module, modelA, modelB, layerA, layerB, stitch_family, label_type, device, init_batch_num, batch_bound, epochs)

    '''
    # dothething
    ...

    mlflow.log_metric("loss", loss.item(), step=step)

    info = {"state_dict": modelAxB.state_dict(),
            "opt": optimizer.state_dict(),
            "metadata": ...}
    save_as_artifact(info, Path("weights") / "modelAxB.pt")'''



if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    args = parser.parse_args()


    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
    mlflow.set_experiment("learnable--stitching")

    with mlflow.start_run(run_name=f"my fancy run {args.modelA}"):
        mlflow.log_params(vars(args))
        main(**vars(args))