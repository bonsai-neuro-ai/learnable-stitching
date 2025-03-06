import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nn_lib.models.utils
import torch
from nn_lib.analysis.stitching import Conv1x1StitchingLayer, create_stitching_model
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import get_pretrained_model, graph_utils
from torch.fx import symbolic_trace, GraphModule
from tqdm.auto import tqdm

import seaborn #for beatiful graphs
import pandas as pd

#Set torch to use cuda GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset root directory, 
#sShoulnd't it just be datasets? There is no data folder in the src directory?
#what is number of workers? 
data_module = ImageNetDataModule(root_dir="/data/datasets", batch_size = 100, num_workers = 10)

#prepare and setup data
data_module.prepare_data()
data_module.setup("test")

#Get pretrained models that will be stitched together
modelA = get_pretrained_model("resnet18")
modelB = get_pretrained_model("resnet34")

layersA = modelA.named_children()
layersB = modelB.named_children()

#Get the graph trace of our pretrained models 
modelA = symbolic_trace(modelA)
modelB = symbolic_trace(modelB)

# single conv operations
#Why do we need to do this? 
modelA = graph_utils.squash_all_conv_batchnorm_pairs(modelA)
modelB = graph_utils.squash_all_conv_batchnorm_pairs(modelB)

# Set up models on the CUDA gpus and put them in eval mode
modelA = modelA.to(device).eval()
modelB = modelB.to(device).eval()

#Define useful function
def quick_run_and_check(model, images, labels, name):
    with torch.no_grad():
        output = model(images)
    score = torch.sum(torch.argmax(output, dim=1) == labels).item() / len(labels)
    print(f"{name}: {torch.sum(torch.argmax(output, dim=1) == labels).item()} / {len(labels)}")
    return score

#create stitching model, train the stitching layer, record loss value, repeat for all different possible connections 
#how can I more easily iterate across layers? Across nodes? across different layer sections? How to pull out adds?
#how to stitch between different sections? 
modelA.graph.print_tabular()

#get split points between layers, could be much more efficient
layersA = []
layersB = []

for node in modelA.graph.nodes: #All the stitch points for modelA
    if (node.name.count("add") > 0):
        layersA.append(node.name)

for node in modelB.graph.nodes: #all the stitch points for modelB
    if (node.name.count("add") > 0):
        layersB.append(node.name)

score = {"name": [], "value": [], "ParentA":  [], "ParentB": []}
for layerA in layersA:
    for layerB in layersB:

        #use dummy inputs to extract input and out put shape for desired layers
        dummy_input = torch.zeros(data_module.shape, device=device).unsqueeze(0)
        dummy_activationsA = graph_utils.get_subgraph(modelA, inputs=["x"], output=layerA)(dummy_input)
        dummy_activationsB = graph_utils.get_subgraph(modelB, inputs=["x"], output=layerB)(dummy_input)
        

            #create stitched model
        stitching_layer = Conv1x1StitchingLayer(
            from_shape=dummy_activationsA.shape[1:],
            to_shape=dummy_activationsB.shape[1:],
        )


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
plot.figure.savefig("practice_plot.png")