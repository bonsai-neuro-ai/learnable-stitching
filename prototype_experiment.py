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
from torch.nn import Conv2d, functional
from tqdm.auto import tqdm


#from stitching_demo import quick_run_and_check, display_model_graph

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
stitching_families = {"conv1x1": Conv2d}

#Knob 3 
#The selection of the loss function
loss_functions = {"class_labels": 0, "soft_labels": 1}

#Set torch to use cuda GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset root directory, 
#sShoulnd't it just be datasets? There is no data folder in the src directory?
#what is number of workers? 
data_module = datalist["imagenet"]

#prepare and setup data
data_module.prepare_data()
data_module.setup("test")

#Get pretrained models that will be stitched together
modelA = modellist["imagenet"]["resnet50"]
modelB = modellist["imagenet"]["resnet101"]

#Get the graph trace of our pretrained models 
traceA = symbolic_trace(modelA)
traceB = symbolic_trace(modelB)
modelA = gmp.GraphModulePlus(root = traceA, graph = traceA.graph, class_name = "HK-Arson")
modelB = gmp.GraphModulePlus(root = traceB, graph = traceB.graph, class_name = "HK-Butcher")




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

#display_model_graph(modelA, "temp.png")

#create stitching model, train the stitching layer, record loss value, repeat for all different possible connections 
#how can I more easily iterate across layers? Across nodes? across different layer sections? How to pull out adds?
#how to stitch between different sections? 
#modelA.graph.print_tabular()


score = {"name": [], "value": [], "ParentA":  [], "ParentB": []}
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
        d_a.to(device)
        d_b.to(device)
        
        #print(dummy_activationsA.shape[1:] == d_a.shape[1:])
        #print(dummy_activationsB.shape[1:] == d_b.shape[1:])

            #create stitched model
    
        linear_layer = torch.nn.Linear(
            in_features= d_a.shape[1],
            out_features=d_b.shape[1]
        )

        ''' dead idea to try and get nn.sequential to work
        class stitch_resize(nn.Module):
            def __init__(
                self,
                kernal_size = 1,
                stride = 1,
                dilation = 1,
                padding = 0
            ):
                super().__init__()
                self._resize_kwargs = {
                    "size": size,
                    "kernal_size": kernal_size,
                    "stride": stride,
                    "dilation": dilation,
                    "padding": padding
                }

            def forward(self, x):
                return Resize(size=conv2d_shape_inverse(out_shape=d_b.shape[-2:], **self._resize_kwargs))'''


        #select the in_features and out_features to be the channel dimension of the convolution dummy inputs, probably not neccessary
        #nn.sequential requires all inputs to be nn.module, thus cannot use resize, 
        mode = "bilinear"
        stitching_layer = nn.Sequential(
            fl.Interpolate2d(size=conv2d_shape_inverse(out_shape=d_b.shape[-2:], kernel_size=1, stride = 1, dilation = 1, padding = 0), mode = mode),
            (fl.RegressableConv2d(in_channels=d_a.shape[1], out_channels=d_b.shape[1], kernel_size=1))
        )

        #stitching_layer.add_module('inter', fl.Interpolate2d(size=conv2d_shape_inverse(out_shape=d_b.shape[-2:], kernel_size=1, stride = 1, dilation = 1, padding = 0), mode = mode))
        
        
        '''
        for name, module in stitching_layer.named_children():
            print(name)
            for n, m in module.named_children():
                print(name, n)
        
        #code to get a graph of the stitching layer, must remove regressableConv2d to use symbolic trace
        stitch_graph = symbolic_trace(stitching_layer)
        stitch_graph.graph.inserting_after()
        #stitch_graph.graph.call_module("inter", args=(), kwargs={})
        display_model_graph(stitch_graph, "Stitch_graph.png") #used to retrieve a snippet of the potential graph of the stitching layer to see the input node
        '''



        #stitching_layer = symbolic_trace(stitching_layer)
        stitching_layer.to(device)

 
        assert stitching_layer(d_a).shape == d_b.shape
        display_model_graph(modelA_, "DonationA_Fig.png")
        display_model_graph(modelB_, "DonationB_Fig.png")

        #Stitch the stitching layer to the bottom of A_ subgraph
        #modelAx = modelA_.new_from_merge({ "DonationA": modelA_, "": stitching_layer}, {"DonationA_"+layerA: [".0"]}, auto_trace = False)

        #Stich the half made stitchign modelA to the modelB, get complete stitching model
        #without the stitching layer, we cannot recompile
        #modelAxB = modelA_.new_from_merge({"DonationA": modelA_, "DonationB": modelB_},
         #                                {"DonationB_"+layerB: ["DonationB_"+layerA]})
        

        #display_model_graph(modelAx, "Stitched_Model_fig.png")

        #freeze context for training the stitching layer
        #with nn_lib.models.utils.frozen(modelA, modelB):

        #freeze context for downstream learning
        #with nn_lib.models.utils.frozen(modelA):
        
        '''
        modelAB = graph_utils.stitch_graphs(
            named_modules={
                "modelA": modelA,
                "stitching_layer": stitching_layer,
                "modelB": modelB,
            },
            rewire_layers_from_to={
                "modelA_" + layerA: "stitching_layer_x",
                "stitching_layer_conv1x1": "modelB_" + layerB,
            },
            input_names=["modelA_" + node.name for node in graph_utils.get_inputs(modelA.graph)],
            output_name="modelB_" + graph_utils.get_output(modelB.graph).args[0].name,
        )


        #THis functions is not working, error "expecting 3 channles for input, given 64 chanelles"
        #modelAB = create_stitching_model(model1= modelA, layer1= layerA, input_shape1=dummy_activationsA.shape[1:],
         #                               model2= modelB, layer2= layerB, input_shape2=dummy_activationsB.shape[1:])
        
        # Inject the correct object back in.
        modelAB.stitching_layer = stitching_layer

        # Cleanup the graphs. This for instance removes the modelA outputs and modelB inputs,
        modelAB.delete_all_unused_submodules()


        #Set stitched model onto gpu
        modelAB = modelAB.to(device).eval()

        #Load data onto gpu
        data_loader = data_module.test_dataloader()
        images, labels = next(iter(data_loader))
        images, labels = images.to(device), labels.to(device)

        #Training the stitching layer to hard labels
        print("\nLayerA: " + layerA + " LayerB: " + layerB)
        print("=== DOING REGRESSION INIT ===")
        batch_repsA = graph_utils.get_subgraph(modelA, inputs=["x"], output=layerA)(images)
        batch_repsB = graph_utils.get_subgraph(modelB, inputs=["x"], output=layerB)(images)
        modelAB.stitching_layer.init_by_regression(batch_repsA, batch_repsB)
       
        #sets of runs to see if it is working
        
        quick_run_and_check(modelA, images, labels, "ModelA")
        quick_run_and_check(modelB, images, labels, "ModelB")

        score["ParentA"].append(layerA)
        score["ParentB"].append(layerB)
        score["name"].append("Model--" + layerA + "--" + layerB)
        score["value"].append(quick_run_and_check(modelAB, images, labels, "ModelAB"))

data = pd.DataFrame.from_dict(score)

compare_scores = data.pivot(index="ParentA", columns="ParentB", values="value")
plot = seaborn.heatmap(compare_scores)
plot.figure.savefig("practice_plot.png")'''