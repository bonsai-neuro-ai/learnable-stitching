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

def record_loss_metrics(modelA, modelB, loss, images, labels, name, step):
    #record loss
    mlflow.log_metric(name + "-loss-AxB", loss.item(), step=step)

    #record loss of donors
    outA = modelA(images)
    lossA = torch.nn.functional.cross_entropy(outA, labels)
    mlflow.log_metric(name + "-loss-A", lossA.item(), step=step)

    outB = modelB(images)
    lossB= torch.nn.functional.cross_entropy(outB, labels)
    mlflow.log_metric(name + "-loss-B", lossB.item(), step=step)

    #record stitching penalities
    mlflow.log_metric(name + "-penalty-A", loss.item() - lossA.item(), step=step)
    mlflow.log_metric(name + "-penalty-B", loss.item() - lossB.item(), step=step)

#A function that records the validation accuracy of modelA, ModelB, and their stiched modelAxB
# Currently there does not seem to be a validation set in the dataset in nn.lib, although imagenet does have a validation set
def record_val_accuarcy(data_module, modelA, modelB, modelAxB, name, step):
    data_module.setup("test")
    data_loader = data_module.test_dataloader()

    accA = 0
    accB = 0
    accAxB = 0

    #calculate the accuarcy for each batch of the evalaution data set
    #the return accurarcy function forces the gradients to be froze, so validation data should not be leaking into the training
    for images, labels in data_loader:
        accA += return_accuracy(modelA, images, labels, name)
        accB += return_accuracy(modelB, images, labels, name)
        accAxB += return_accuracy(modelAxB, images, labels, name)
    
    mlflow.log_metric(name + "-validation-A", accA, step=step)
    mlflow.log_metric(name + "-validation-B", accB, step=step)
    mlflow.log_metric(name + "-validation-AxB", accAxB, step=step)

#A function that records the test accuracy of modelA, ModelB, and their stiched modelAxB
def record_test_accuarcy(data_module, modelA, modelB, modelAxB, name, step):
    data_module.setup("test")
    data_loader = data_module.test_dataloader()

    accA = 0
    accB = 0
    accAxB = 0

    #calculate the accuarcy for each batch of the evalaution data set
    #the return accurarcy function forces the gradients to be froze, so validation data should not be leaking into the training
    for images, labels in data_loader:
        accA += return_accuracy(modelA, images, labels, name)
        accB += return_accuracy(modelB, images, labels, name)
        accAxB += return_accuracy(modelAxB, images, labels, name)
    
    mlflow.log_metric(name + "-test-A", accA, step=step)
    mlflow.log_metric(name + "-test-B", accB, step=step)
    mlflow.log_metric(name + "-test-AxB", accAxB, step=step)



def create_and_record_stitched_model(data_module, modelA, modelB, layerA, layerB, stitch_family, label_type, device,  init_batch_num :int, batch_bound :int, epochs :int):
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

        #return_accuracy(modelA, images, labels, "ModelA")
        #return_accuracy(modelB, images, labels, "ModelB")
        #return_accuracy(modelAxB, images, labels, "ModelAxB")
    
    #applying init by regression to speed up training time
    stitching_layer[1].init_by_regression(torch.cat(reg_input), torch.cat(reg_output))
    
    #track steps for mlflow logger
    step = 0

    #freeze context for training the stitching layer
    with frozen(modelA, modelB):
        
        for epoch in tqdm(range(epochs), desc = "Stitching Layer Training over Epochs"): #iterate over the number of desired epochs
                
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

                    
                    #todo: finishing implementing soft labels
                    if label_type == "soft": #soft labels
                        labels = nn.Softmax(modelA(images))

                        
                    # This is task loss, but could be updated to be soft-labels to optimize match to model B
                    loss = torch.nn.functional.cross_entropy(output, labels)

                    #backprop and adjust
                    loss.backward()
                    optimizer.step()

                    #return_accuracy(modelAxB, images, labels, "ModelAxB") #sanity check to see if AxB improves 

                    record_loss_metrics(modelA, modelB, loss, images, labels, "Stitching", step)
                    step+= 1

            #save modelAxB state each epoch, as well as the state of the optimizer
            info = {"state_dict": modelAxB.state_dict(),
                    "opt": optimizer.state_dict(),
                    "metadata": ...}
            save_as_artifact(info, Path("weights") / "stitching-modelAxB.pt", run_id=None)


            '''
            #I left this uncommented because I am unsure if it is properly completed or needed
            #Uncomment if you want validation accuracy data, but be warned I have not gotten to test it
            #record the validation accuracy at every epoch
            #also could use this validation accuracy to choose when to stop training 
            record_val_accuarcy(data_module, modelA, modelB, modelAxB, name = "Stitching", step = step)
            '''

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
            
        for epoch in tqdm(range(epochs), desc = "Downstream Learning over Epochs"): #iterate over the number of desired epochs

            modelAxB.donorB.train()
            optimizer = torch.optim.Adam(modelAxB.parameters(), lr=0.000000001)
                            
            for i, (images, labels) in tqdm(enumerate(data_loader), total=batch_bound, desc= "Downstrearm Learning"): #iterate over an epoch
            #for i, (images, labels) in tqdm(enumerate(zip(images_2,labels_2)), total=len(images_2), desc= "Downstream Learning"): #same images and labels used for the stitching layer
                if (i == batch_bound): 
                    break

                images = images.to(device)
                labels = labels.to(device)
                        
                optimizer.zero_grad()
                output = modelAxB(images)
                #print(labels)

                #todo: finishing implementing soft labels
                if label_type == "soft": #soft labels can be derrived by picking the largest value from the softmax of the logits
                    labels = nn.Softmax(modelA(images))

                                    
                # This is task loss, but could be updated to be soft-labels to optimize match to model B
                loss = torch.nn.functional.cross_entropy(output, labels)

                
                #backprop and adjust
                loss.backward()
                optimizer.step()       

                #return_accuracy(modelB, images, labels, "ModelB") #sanity check to see if AxB improves 
                #return_accuracy(modelAxB, images, labels, "ModelAxB") #sanity check to see if AxB improves 

                #record loss
                record_loss_metrics(modelA, modelB, loss, images, labels, "Downstream", step)
                step += 1

            #save modelAxB state each epoch, as well as the state of the optimizer
            info = {"state_dict": modelAxB.state_dict(),
                    "opt": optimizer.state_dict(),
                    "metadata": ...}
            save_as_artifact(info, Path("weights") / "DownsteamLearning-modelAxB.pt", run_id=None)

            '''
            #I left this uncommented because I am unsure if it is properly completed or needed
            #Uncomment if you want validation accuracy data, but be warned I have not gotten to test it
            #record the validation accuracy at each epoch
            #also could use this validation accuracy to choose when to stop training 
            record_val_accuarcy(data_module, modelA, modelB, modelAxB, name = "Stitching", step = step)
            '''
    
def main(dataset: str, modelA: str, modelB: str, layerA: str, layerB: str, stitch_family :str, label_type: str, epochs: int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_batch_num = 5
    batch_bound = -100 #no bound on the number of batches, training will do a full epoch

    #todo: implement COCO segementation 
    data_module = ImageNetDataModule(root_dir="/data/datasets", batch_size = 100, num_workers = 10) #dataset is currently hardcoded to be imagenet
    modelA = get_pretrained_model(modelA)
    modelB = get_pretrained_model(modelB)
    layerA = layerA
    layerB = layerB

    stitch_family = "1x1Conv" #currently will not do anything 
    label_type = "class" #currently hard coded to be class labels as soft labels are currently unsupported

    #todo implement soft labels

    epochs = epochs

    create_and_record_stitched_model(data_module, modelA, modelB, layerA, layerB, stitch_family, label_type, device, init_batch_num, batch_bound, epochs)



if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    args = parser.parse_args()


    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
    mlflow.set_experiment("learnable--stitching")
    
    #print(f"{args.dataset}--{args.modelA}-{args.layerA}--x{args.stitch_family}x--{args.modelB}-{args.layerB}--label_{args.label_type}")

    with mlflow.start_run(run_name=f"{args.dataset}--{args.modelA}-{args.layerA}--x{args.stitch_family}x--{args.modelB}-{args.layerB}--label_{args.label_type}"):
        mlflow.log_params(vars(args))
        main(**vars(args))