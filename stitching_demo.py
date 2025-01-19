import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nn_lib.models.utils
import torch
from nn_lib.analysis.stitching import Conv1x1StitchingLayer
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import get_pretrained_model, graph_utils
from torch.fx import symbolic_trace, GraphModule
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset, pointing root_dir to the location on the server where we keep shared datasets
data_module = ImageNetDataModule(root_dir="/data/datasets/", batch_size=100, num_workers=10)
# DataModule is a concept from pytorch lightning. For performance reasons, data modules are lazy
# and wait to load the data until we actually ask for it. We need to tell the data module to
# actually run its setup routines. There's room for API improvement here, especially the unexpected
# behavior where prepare_data() has a side-effect of setting up the default transforms, which are
# then required by setup().
data_module.prepare_data()
data_module.setup("test")

# Model names must be recognized by torchvision.models.get_model
modelA = get_pretrained_model("resnet18")
modelB = get_pretrained_model("resnet34")

# To do stitching or other 'model surgery', we need to convert from standard nn.Module objects
# to torch's GraphModule objects. This is done by 'tracing' the input->output flow of the model
modelA = symbolic_trace(modelA)
modelB = symbolic_trace(modelB)

assert isinstance(modelA, GraphModule)
assert isinstance(modelB, GraphModule)

# Since the models were pretrained already, we'll also squash all conv+batchnorm operations into
# single conv operations
modelA = graph_utils.squash_all_conv_batchnorm_pairs(modelA)
modelB = graph_utils.squash_all_conv_batchnorm_pairs(modelB)

# Final step of model setup: put them on device and in eval mode
modelA = modelA.to(device).eval()
modelB = modelB.to(device).eval()

# Print out the layer names of the model; these are the names of the nodes in the computation graph
modelA.graph.print_tabular()

# We can also visualize the model as a graph
def display_model_graph(mdl, dpi=200):
    image = graph_utils.to_dot(mdl.graph).create_png(prog="dot")
    with io.BytesIO(image) as f:
        image = mpimg.imread(f)
    plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

display_model_graph(modelA)

# Print some metadata about each model's inputs and outputs. This illustrates how the GraphModule
# contains a Graph which then contains node and connection information between the layers.
print("=== Metadata for ModelA ===")
print("Inputs:", [node.name for node in graph_utils.get_inputs(modelA.graph)])
print("Output:", graph_utils.get_output(modelA.graph).name)
print("=== Metadata for ModelB ===")
print("Inputs:", [node.name for node in graph_utils.get_inputs(modelB.graph)])
print("Output:", graph_utils.get_output(modelB.graph).name)

# Create a hybrid stitched model. See the implementation of `create_stitching_model()` for
# details. The main idea is to (1) run some dummy input through each model to query the shapes of
# the tensors at the desired layers; (2) instantiate a Conv1x1StitchingLayer from those shapes; (
# 3) call graph_utils.stitch_graphs() (which is a slightly more generic 'rewiring' interface I
# wrote) to do the model surgery; and (4) cleanup the graphs. Some of this is a little finicky,
# so there is a nn_lib.analysis.stitching.create_stitching_model function that does steps (1)
# thru (4) in one go. Maybe we can clean up this API a bit.
layerA = "add_3"
layerB = "add_5"

# Step (1): Get 'subgraph' models which go input -> desired layer. Then, look at the output shape
# of these sub-models to determine the shape of the tensors at the desired layers. Note: if the two
# models were trained on different datasets or expect different input sizes, this needs to be
# modified.
dummy_input = torch.zeros(data_module.shape, device=device).unsqueeze(0)
dummy_activationsA = graph_utils.get_subgraph(modelA, inputs=["x"], output=layerA)(dummy_input)
dummy_activationsB = graph_utils.get_subgraph(modelB, inputs=["x"], output=layerB)(dummy_input)

# Step (2): Create a Conv1x1StitchingLayer from the shapes of the tensors at the desired layers
stitching_layer = Conv1x1StitchingLayer(
    from_shape=dummy_activationsA.shape[1:],
    to_shape=dummy_activationsB.shape[1:],
)

# Step (3): model surgery. This works by specifying a dict of modules, then a dict saying which
# layer from which module will be wired into which layer from which other module. Note that all
# layer names are identified by "{module_name}_{layer_name}" where the module_name comes from the
# named_modules dict we pass in. In the example, we are wiring "modelA_add_3" into the input of
# the stitching layer, and GraphModules always have a default input named 'x'. The output of this
# stitching layer happens to be a layer called 'conv1x1', which we wire into "modelB_add_5".
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

# Bugfix: stitch_graphs() creates a 'stitching_layer' attribute, but it has the wrong class. Inject
# the correct object back in.
modelAB.stitching_layer = stitching_layer

# Step (4): Cleanup the graphs. This for instance removes the modelA outputs and modelB inputs,
# 'trimming' the resulting computation graph just to [modelApart1 -> stitching_layer -> modelBpart2]
# Nicely enough, this is built into the GraphModule class as a method:
modelAB.delete_all_unused_submodules()

# Also ensure the stitched model is on-device and in eval mode:
modelAB = modelAB.to(device).eval()

# Let's also visualize the stitched model as an image
display_model_graph(modelAB)

# Sanity-check that we can run all 3 models
data_loader = data_module.test_dataloader()
images, labels = next(iter(data_loader))
images, labels = images.to(device), labels.to(device)
print(f"Sanity-checking on a single test batch containing {len(images)} images")

def quick_run_and_check(model, images, labels, name):
    with torch.no_grad():
        output = model(images)
    print(f"{name}: {torch.sum(torch.argmax(output, dim=1) == labels).item()} / {len(labels)}")


quick_run_and_check(modelA, images, labels, "ModelA")
quick_run_and_check(modelB, images, labels, "ModelB")
# we expect this one to be bad because the stitching layer was only randomly initialized
quick_run_and_check(modelAB, images, labels, "ModelAB")

# Record parameters before any training so that we can sanity-check that only the correct things
# are changing. The .clone() is necessary so we get a snapshot of the parameters at this point in
# time, rather than a reference to the parameters which will be updated later.
paramsA = {k: v.clone() for k, v in modelA.named_parameters()}
paramsB = {k: v.clone() for k, v in modelB.named_parameters()}
paramsAB = {k: v.clone() for k, v in modelAB.named_parameters()}

# Now that we have a functioning stitched model, let's update the stitching layer. First way to do
# this is with the regression-based method. Note that this will in general be better if we use a
# bigger batch.
print("=== DOING REGRESSION INIT ===")
batch_repsA = graph_utils.get_subgraph(modelA, inputs=["x"], output=layerA)(images)
batch_repsB = graph_utils.get_subgraph(modelB, inputs=["x"], output=layerB)(images)
modelAB.stitching_layer.init_by_regression(batch_repsA, batch_repsB)

# Assert that no parameters changed *except* for modelAB.stitching_layer
for k, v in modelA.named_parameters():
    assert torch.allclose(v, paramsA[k])
for k, v in modelB.named_parameters():
    assert torch.allclose(v, paramsB[k])
for k, v in modelAB.named_parameters():
    if k.startswith("stitching_layer"):
        assert not torch.allclose(v, paramsAB[k])
    else:
        assert torch.allclose(v, paramsAB[k])

# Re-run and see if it's improved
quick_run_and_check(modelA, images, labels, "ModelA")
quick_run_and_check(modelB, images, labels, "ModelB")
quick_run_and_check(modelAB, images, labels, "ModelAB")

# For fine-tuning the stitching layer while freezing the rest of the model, I've created a model
# freezing context manager. This also illustrates an important design choice: the parameters of
# modelAB are *shared* with the original modelA and modelB models. Two important consequences of
# this: (1) we can freeze modelA while training modelAB, and the parameters of modelA will not be
# updated; (2) if we update the parameters of modelAB, the parameters of modelA and modelB will
# also be updated. We need to always be careful to make copies of models if we want to avoid this.
data_module.setup("fit")
train_dataloader = data_module.train_dataloader()
history = []
# To train stitching layer AND downstream model, just remove 'modelB' from the list of frozen models
with nn_lib.models.utils.frozen(modelA, modelB):
    modelAB.stitching_layer.train()
    optimizer = torch.optim.Adam(modelAB.parameters(), lr=0.001)
    # Train for 100 steps or 1 epoch, whichever comes first
    for step, (im, la) in tqdm(enumerate(train_dataloader), total=100, desc="Train Stitching Layer"):
        optimizer.zero_grad()
        output = modelAB(images)
        # This is task loss, but could be updated to be soft-labels to optimize match to model B
        loss = torch.nn.functional.cross_entropy(output, labels)
        history.append(loss.item())
        loss.backward()
        optimizer.step()

        step += 1
        if step == 100:
            break

plt.plot(history)
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training the Stitching Layer by itself")
plt.show()

# Assert that no parameters changed *except* for modelAB.stitching_layer
for k, v in modelA.named_parameters():
    assert torch.allclose(v, paramsA[k])
for k, v in modelB.named_parameters():
    assert torch.allclose(v, paramsB[k])
for k, v in modelAB.named_parameters():
    if k.startswith("stitching_layer"):
        assert not torch.allclose(v, paramsAB[k])
    else:
        assert torch.allclose(v, paramsAB[k])

# Re-run and see if it's improved
quick_run_and_check(modelA, images, labels, "ModelA")
quick_run_and_check(modelB, images, labels, "ModelB")
quick_run_and_check(modelAB, images, labels, "ModelAB")
