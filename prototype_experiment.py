import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nn_lib.models.utils
import torch
import torch.nn as nn



import nn_lib.models.graph_module_plus as gmp
import nn_lib.models.fancy_layers as fl

from torchvision.transforms import Resize
from nn_lib.models.utils import conv2d_shape_inverse
#from nn_lib.analysis.similarity import Conv1x1StitchingLayer, create_stitching_model
from nn_lib.datasets import ImageNetDataModule, CocoDetectionDataModule
from nn_lib.models import get_pretrained_model, graph_utils
from torch.fx import symbolic_trace, GraphModule, graph
#from torch.nn import Conv2d, functional
from tqdm.auto import tqdm


#from stitching_demo import quick_run_and_check

import seaborn #for beatiful graphs
import pandas as pd

#Knob #1
#independent variable is the dataset
imagenet = ImageNetDataModule(root_dir="/data/datasets", batch_size = 100, num_workers = 10)
coco_seg = CocoDetectionDataModule(root_dir="/data/datasets", batch_size = 100, num_workers = 10)
datalist = {"imagenet": imagenet, "coco": coco_seg}

#Knob #1.5
#The selection of the stitching parents from the pretrained models for the given dataset
imagenet_models = {"resnet50": get_pretrained_model("resnet18"), "resnet101": get_pretrained_model("resnet34"), "ViT": get_pretrained_model("vit_b_32")}
coco_models = {}
modellist = {"imagenet": imagenet_models, "coco": coco_models}

#Knob 2
#The selection of the stitching layer


#Knob 3 
#The selection of the loss function
loss_functions = ["class", "soft"]
label_type = loss_functions[0]


#Set torch to use cuda GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset root directory, 
#sShoulnd't it just be datasets? There is no data folder in the src directory?
#what is number of workers? 
data_module = datalist["imagenet"]
init_batch_num = 1
batch_bound = 100


#prepare and setup data
data_module.prepare_data()


#Get pretrained models that will be stitched together
modelA = modellist["imagenet"]["resnet50"]
modelB = modellist["imagenet"]["resnet101"]

#Get the graph trace of our pretrained models 
traceA = symbolic_trace(modelA)
traceB = symbolic_trace(modelB)
modelA = gmp.GraphModulePlus(root = traceA, graph = traceA.graph, class_name = "mA")
modelB = gmp.GraphModulePlus(root = traceB, graph = traceB.graph, class_name = "mB")




# single conv operations
#Why do we need to do this? 
modelA = modelA.squash_all_conv_batchnorm_pairs()
modelB = modelB.squash_all_conv_batchnorm_pairs()

#get split points between layers, could be much more efficient
layersA = []
layersB = []

for node in modelA.graph.nodes: #All the stitch points for modelA
    if (node.name.count("add") > 0):
        layersA.append(node.name)

for node in modelB.graph.nodes: #all the stitch points for modelB
    if (node.name.count("add") > 0):
        layersB.append(node.name)

layersA = [layersA[2]]
layersB = [layersB[4]]

# Set up models on the CUDA gpus and put them in eval mode
modelA = modelA.to(device).eval()
modelB = modelB.to(device).eval()

def display_model_graph(graph, file, dpi=200):
    image = graph.to_dot().create_png(prog="dot")
    with io.BytesIO(image) as f:
        image = mpimg.imread(f)
    plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(file)
    plt.show()

def quick_run_and_check(model, images, labels, name):
    with torch.no_grad():
        output = model(images)
    print(f"{name}: {torch.sum(torch.argmax(output, dim=1) == labels).item()} / {len(labels)}")

#display_model_graph(modelA, "temp.png")


data_module.setup("test")
test_data_loader = data_module.test_dataloader()
test_images, test_labels = next(iter(test_data_loader))
test_images, test_labels = test_images.to(device), test_labels.to(device)

data_module.setup("fit")
data_loader = data_module.train_dataloader()




for layerA in layersA:
    for layerB in layersB:
        

        #use dummy inputs to extract input and out put shape for desired layers

        modelA_ = modelA.extract_subgraph(inputs = ["x"], output=layerA)
        #modelB_ = modelB.extract_subgraph(inputs = layerB, output="fc")
        modelB_ = modelB.extract_subgraph(inputs = ["x"], output=layerB)
        modelA_.to(device)
        modelB_.to(device)



        dummy_input = torch.zeros(data_module.shape, device=device).unsqueeze(0)
        #dummy_activationsA = graph_utils.get_subgraph(modelA, inputs=["x"], output=layerA)(dummy_input)
        #dummy_activationsB = graph_utils.get_subgraph(modelB, inputs=["x"], output=layerB)(dummy_input)


        d_a = modelA_(dummy_input)
        d_b = modelB_(dummy_input)
  

        #retrieve next node for stitching the graph
        layerB_next = modelB._resolve_nodes(layerB)[0].users
        layerB_next = next(iter(layerB_next)).name

        
        #print(dummy_activationsA.shape[1:] == d_a.shape[1:])
        #print(dummy_activationsB.shape[1:] == d_b.shape[1:])

        #select the in_features and out_features to be the channel dimension of the convolution dummy inputs, probably not neccessary
        #nn.sequential requires all inputs to be nn.module, thus cannot use resize, 
        mode = "bilinear"
        stitching_layer = nn.Sequential(
            fl.Interpolate2d(size=conv2d_shape_inverse(out_shape=d_b.shape[-2:], kernel_size=1, stride = 1, dilation = 1, padding = 0), mode = mode),
            (fl.RegressableConv2d(in_channels=d_a.shape[1], out_channels=d_b.shape[1], kernel_size=1))
        )

        #stitching_layer.add_module('inter', fl.Interpolate2d(size=conv2d_shape_inverse(out_shape=d_b.shape[-2:], kernel_size=1, stride = 1, dilation = 1, padding = 0), mode = mode))
        stitching_layer.to(device)
        
 
        assert stitching_layer(d_a).shape == d_b.shape
        display_model_graph(modelA_, "DonationA_Fig.png")
        display_model_graph(modelB_, "DonationB_Fig.png")

        #Stitch the stitching layer to the bottom of A_ subgraph
        modelAxB = modelA_.new_from_merge({ "donorA": modelA, "sl": stitching_layer, "donorB": modelB}, 
                                          {"sl": ["donorA_"+layerA], "donorB_"+layerB_next: ["sl"]}, 
                                          auto_trace = False)


        display_model_graph(modelAxB, "Stitched_Model_fig.png")

        # Record parameters before any training so that we can sanity-check that only the correct things
        # are changing. The .clone() is necessary so we get a snapshot of the parameters at this point in
        # time, rather than a reference to the parameters which will be updated later.
        paramsA = {k: v.clone() for k, v in modelA.named_parameters()}
        paramsB = {k: v.clone() for k, v in modelB.named_parameters()}
        paramsAxB = {k: v.clone() for k, v in modelAxB.named_parameters()}



        reg_input = []
        reg_output = []
        #initialize the stitchig layer using regression for a training speed up
        #initialize a couple of batches according to the init_batch 
        for i, (images, labels) in tqdm(enumerate(data_loader), total= init_batch_num, desc = "collecting batches for regression init"):
            if (i >= init_batch_num): 
                break

            images = images.to(device)
            labels = labels.to(device)

            reg_input.append(stitching_layer[0](modelA_(images)))
            reg_output.append(modelB_(images))

        #sanity print statements to see the initial accuracy of the models
        quick_run_and_check(modelA, test_images, test_labels, "ModelA")
        quick_run_and_check(modelB, test_images, test_labels, "ModelB")
        quick_run_and_check(modelAxB, test_images, test_labels, "ModelAxB")
        
        #Running Init By Regression
        stitching_layer[1].init_by_regression(torch.cat(reg_input), torch.cat(reg_output))

        #sanity print statements to see the initial accuracy of the models
        print("INIT_BY_REGRESSION")
        quick_run_and_check(modelA, test_images, test_labels, "ModelA")
        quick_run_and_check(modelB, test_images, test_labels, "ModelB")
        quick_run_and_check(modelAxB, test_images, test_labels, "ModelAxB")

        #freeze context for training the stitching layer
        with nn_lib.models.utils.frozen(modelA, modelB):
            
            modelAxB.sl.train()
            optimizer = torch.optim.Adam(modelAxB.parameters(), lr=0.000001)

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

        #sanity print statements to observe stitching layer improvement
        quick_run_and_check(modelA, test_images, test_labels, "ModelA")
        quick_run_and_check(modelB, test_images, test_labels, "ModelB")
        quick_run_and_check(modelAxB, test_images, test_labels, "ModelAxB")

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
        with nn_lib.models.utils.frozen(modelA, stitching_layer):
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
        
            # Assert that no parameters changed *except* for modelB
            for k, v in modelA.named_parameters():
                assert torch.allclose(v, paramsA[k])
            for k, v in modelAxB.named_parameters():
                if k.startswith("stitching_layer"):
                    assert torch.allclose(v, paramsAxB[k])

        modelAxB.donorB.eval()

        #sanity print statements to observe downstream learning is working properly
        quick_run_and_check(modelA, test_images, test_labels, "ModelA")
        quick_run_and_check(modelB, test_images, test_labels, "ModelB")
        quick_run_and_check(modelAxB, test_images, test_labels, "ModelAxB")