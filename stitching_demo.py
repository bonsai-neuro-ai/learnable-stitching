import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nn_lib.models.utils
import torch
<<<<<<< HEAD
from nn_lib.analysis.stitching import Conv1x1StitchingLayer, create_stitching_model
=======
from torch import nn
import torch.nn.functional as F
>>>>>>> ca87031dc423bae092bb5d87066ccae9c6b98e59
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import get_pretrained_model, GraphModulePlus
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv1x1StitchingLayer(nn.Module):
    def __init__(
        self,
        from_shape: tuple[int, int, int],
        to_shape: tuple[int, int, int],
    ):
        super().__init__()

        self.from_shape = from_shape
        self.to_shape = to_shape
        self.conv1x1 = nn.Conv2d(
            in_channels=from_shape[0],
            out_channels=to_shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def maybe_resize(self, x):
        if self.from_shape[1:] != self.to_shape[1:]:
            return F.interpolate(x, size=self.to_shape[1:], mode="bilinear", align_corners=False)
        else:
            return x

    def forward(self, x):
        return self.conv1x1(self.maybe_resize(x))

    @torch.no_grad()
    def init_by_regression(
        self, from_data: torch.Tensor, to_data: torch.Tensor, check_sanity: bool = False
    ):
        # Ensure that the input data has the correct shape
        c1, h1, w1 = self.from_shape
        c2, h2, w2 = self.to_shape
        batch = from_data.shape[0]
        if batch != to_data.shape[0]:
            raise ValueError(
                f"from_data has batch size {batch}, " f"to_data has batch size {to_data.shape[0]}"
            )
        if from_data.shape[1:] != (c1, h1, w1):
            raise ValueError(f"from_data has shape {from_data.shape[1:]}, expected {(c1, h1, w1)}")
        if to_data.shape[1:] != (c2, h2, w2):
            raise ValueError(f"to_data has shape {to_data.shape[1:]}, expected {(c2, h2, w2)}")

        from_data = self.maybe_resize(from_data)

        # Reshape from (batch, channel, height, width) to (batch*height*width, channel)
        from_data_flat = from_data.permute(0, 2, 3, 1).reshape(batch * h2 * w2, c1)
        to_data_flat = to_data.permute(0, 2, 3, 1).reshape(batch * h2 * w2, c2)

        # Perform linear regression including a column of ones for the bias term
        from_data_flat = torch.cat([from_data_flat, torch.ones_like(from_data_flat[:, :1])], dim=1)
        weights = torch.linalg.lstsq(from_data_flat, to_data_flat).solution
        # To copy reshaped weights back to the conv1x1 layer, we need to include a clone() call,
        # otherwise we'll get errors related to memory strides.
        self.conv1x1.weight.data = weights[:-1].T.reshape(self.conv1x1.weight.shape).clone()
        self.conv1x1.bias.data = weights[-1].clone()

        if check_sanity:
            # Sanity check that reshaping did what we think
            pred_flat = (from_data_flat @ weights).reshape(batch, h2, w2, c2).permute(0, 3, 1, 2)
            pred_conv = self.conv1x1(from_data)

            # V2 if we didn't transpose the weights earlier
            self.conv1x1.weight.data = weights[:-1].reshape(self.conv1x1.weight.shape)
            self.conv1x1.bias.data = weights[-1]
            pred_conv_2 = self.conv1x1(from_data)

            correlations = torch.corrcoef(
                torch.stack([to_data.flatten(), pred_conv.flatten(), pred_conv_2.flatten()], dim=0)
            )
            print(f"Correlation (data, prediction) with transpose: {correlations[0, 1]}")
            print(f"Correlation (data, prediction) without transpose: {correlations[0, 2]}")

            diff = torch.abs(pred_flat - pred_conv)
            print(f"Max abs difference (flat pred - conv pred): {diff.max()}")
            print(
                f"Max relative difference (flat pred - conv pred) / flat pred:"
                f"{diff.max() / pred_flat.abs().max()}"
            )

            assert torch.allclose(
                pred_flat, pred_conv, atol=0.01, rtol=0.001
            ), "Linear regression sanity-check failed"

    def __repr__(self):
        return f"Conv1x1StitchingLayer(from_shape={self.from_shape}, to_shape={self.to_shape})"

    def __str__(self):
        return self.__repr__()


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
modelA = GraphModulePlus.new_from_trace(modelA).squash_all_conv_batchnorm_pairs().to(device).eval()
modelB = GraphModulePlus.new_from_trace(modelB).squash_all_conv_batchnorm_pairs().to(device).eval()

# Print out the layer names of the model; these are the names of the nodes in the computation graph
modelA.graph.print_tabular()


# We can also visualize the model as a graph
def display_model_graph(mdl, dpi=200):
    image = mdl.to_dot().create_png(prog="dot")
    with io.BytesIO(image) as f:
        image = mpimg.imread(f)
    plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("TempGrap.png")
    plt.show()


display_model_graph(modelA)
display_model_graph(modelB)

<<<<<<< HEAD
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
layerA = "add_1"
=======
# Create a hybrid stitched model.
layerA = "add_3"
>>>>>>> ca87031dc423bae092bb5d87066ccae9c6b98e59
layerB = "add_5"
layerB_next = modelB._resolve_nodes(layerB)[0].users
layerB_next = next(iter(layerB_next)).name
print("After", layerB, "comes", layerB_next)

# Step (1): Get 'subgraph' models which go input -> desired layer. Then, look at the output shape
# of these sub-models to determine the shape of the tensors at the desired layers. Note: if the two
# models were trained on different datasets or expect different input sizes, this needs to be
# modified.
dummy_input = torch.zeros(data_module.shape, device=device).unsqueeze(0)
embedding_getterA = GraphModulePlus.new_from_copy(modelA).extract_subgraph(output=layerA)
embedding_getterB = GraphModulePlus.new_from_copy(modelB).extract_subgraph(output=layerB)

# Step (2): Create a Conv1x1StitchingLayer from the shapes of the tensors at the desired layers
stitching_layer = Conv1x1StitchingLayer(
    from_shape=embedding_getterA(dummy_input).shape[1:],
    to_shape=embedding_getterB(dummy_input).shape[1:],
)

# Step (3): model merging surgery. Note the types: modelA and modelB are already GraphModules, but
# stitching_layer is a regular nn.Module. Setting auto_trace=False keeps it as a regular nn.Module,
# which is necessary for us to call init_by_regression() on it later.
modelAB = GraphModulePlus.new_from_merge(
    modules={"modelA": modelA, "stitching_layer": stitching_layer, "modelB": modelB},
    rewire_inputs={
        "stitching_layer": "modelA_" + layerA,
        "modelB_" + layerB_next: "stitching_layer",
    },
    auto_trace=False,
).to(device).eval()

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
modelAB.stitching_layer.init_by_regression(embedding_getterA(images), embedding_getterB(images))

# Assert that no parameters changed *except* for stitched_model.stitching_layer
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
# stitched_model are *shared* with the original modelA and modelB models. Two important consequences of
# this: (1) we can freeze modelA while training stitched_model, and the parameters of modelA will not be
# updated; (2) if we update the parameters of stitched_model, the parameters of modelA and modelB will
# also be updated. We need to always be careful to make copies of models if we want to avoid this.
data_module.setup("fit")
train_dataloader = data_module.train_dataloader()
history = []
# To train stitching layer AND downstream model, just remove 'modelB' from the list of frozen models
with nn_lib.models.utils.frozen(modelA, modelB):
    modelAB.stitching_layer.train()
    optimizer = torch.optim.Adam(modelAB.parameters(), lr=0.001)
    # Train for 100 steps or 1 epoch, whichever comes first
    for step, (im, la) in tqdm(
        enumerate(train_dataloader), total=100, desc="Train Stitching Layer"
    ):
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

# Assert that no parameters changed *except* for stitched_model.stitching_layer
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
