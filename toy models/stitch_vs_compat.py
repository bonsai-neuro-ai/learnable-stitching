import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"

# Seed chosen by keyboard-mashing
# # TODO - multiple seeds/runs and show averages
# torch.manual_seed(256713)

# %% Manually construct activations for model A: a 2D xor problem

xy_a = torch.cartesian_prod(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16))
labels = ((xy_a[:, 0] * xy_a[:, 1]) > 0).long()  # XOR labels

# %% Construct activations for model B by defining an invertible function of xy space which will break A into 'tiles'


def a_to_b(xy: torch.Tensor) -> torch.Tensor:
    out = xy.clone()
    x, y = xy[:, 0], xy[:, 1]
    flip_x_sign = torch.logical_and(y < 0, torch.abs(x) < 0.5)
    out[flip_x_sign, 0] *= -1
    return out


# It is its own inverse
b_to_a = a_to_b

xy_b = a_to_b(xy_a)

assert torch.allclose(b_to_a(xy_b), xy_a), "a_to_b and b_to_a are not inverses!"

# %% Quick vis feature spaces

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(xy_a[:, 0], xy_a[:, 1], c=labels * 1.0)
ax[0].set_title("Reps for Model A")
ax[1].scatter(xy_b[:, 0], xy_b[:, 1], c=labels * 1.0)
ax[1].set_title("Reps for Model B")
plt.show()

# %% Models and training and analysis helpers


def new_model_C(w_init):
    # Will be equivalent to readout = dot(w_init, xy)
    w1 = nn.Linear(2, 4)
    w1.weight.data[:] = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    w1.bias.data.zero_()
    w2 = nn.Linear(4, 1)
    w2.weight.data[:] = torch.tensor([w_init[0], -w_init[0], w_init[1], -w_init[1]]).reshape(1, 4)
    w2.bias.data.zero_()
    return nn.Sequential(w1, nn.ReLU(), w2)


def model_predictions_heatmap(model, density=300, range=(-1, +1), ax=None):
    xy = torch.cartesian_prod(torch.linspace(*range, density), torch.linspace(*range, density))
    with torch.no_grad():
        preds = torch.sigmoid(model(xy).squeeze()).reshape(density, density)

    ax = ax or plt.gca()
    im = ax.imshow(
        preds.numpy(), extent=(*range, *range), origin="lower", vmin=0, vmax=1, cmap="PiYG"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="Predicted probability")
    return ax


def accuracy(xy, labels, model):
    with torch.no_grad():
        outputs = model(xy).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        return (preds == labels).float().mean().item()


def penalize_non_rotations(weights: torch.Tensor):
    """Regularizer to softly encourage weights to be rotation-like"""
    # if weights.ndim != 2:
    #     raise ValueError("weights must be a 2D tensor")
    # wtw = weights.T @ weights
    # identity = torch.eye(wtw.shape[0], device=weights.device)
    # return torch.sum((wtw - identity) ** 2)
    return -torch.linalg.slogdet(weights)[1]


def train_stitched(xy, labels, modelC, n_steps=1000, lr=1e-2) -> tuple[list[float], nn.Sequential]:
    stitching_layer = nn.Linear(2, 2)
    hybrid_model = nn.Sequential(stitching_layer, modelC)

    optimizer = optim.SGD(stitching_layer.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    criterion = nn.BCEWithLogitsLoss()

    loss_history = []
    for _ in trange(n_steps, desc="Training stitching layer"):
        optimizer.zero_grad()
        outputs = hybrid_model(xy).squeeze()
        bce_loss = criterion(outputs, labels.float())
        loss = bce_loss + 1e-3 * penalize_non_rotations(hybrid_model[0].weight)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(bce_loss.item())

    return loss_history, hybrid_model


def continue_training_downstream(xy, labels, hybrid_model, n_steps=1000, lr=1e-2):
    # Now training ALL parameters in the hybrid model not just the stitching layer
    optimizer = optim.SGD(hybrid_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    criterion = nn.BCEWithLogitsLoss()

    loss_history = []
    for _ in trange(n_steps, desc="Continuing training downstream"):
        optimizer.zero_grad()
        outputs = hybrid_model(xy).squeeze()
        bce_loss = criterion(outputs, labels.float())
        loss = bce_loss + 1e-3 * penalize_non_rotations(hybrid_model[0].weight)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(bce_loss.item())

    return loss_history, hybrid_model


# %%

initial_readout = np.ones((2,), dtype=np.float32) / np.sqrt(2)
reference_model_C = new_model_C(initial_readout)

# Sanity-check that C(xy) == xy @ w
with torch.no_grad():
    assert torch.allclose(
        reference_model_C(xy_a).squeeze(), xy_a @ torch.tensor(initial_readout).float().squeeze()
    )

# Sanity-check that C has the expected number of trainable parameters
assert sum(p.numel() for p in reference_model_C.parameters() if p.requires_grad) == 8 + 4 + 4 + 1

# %% Do stitching analysis; confirm that B beats A

history_ac, hybrid_ac = train_stitched(xy_a, labels, new_model_C(initial_readout))
history_bc, hybrid_bc = train_stitched(xy_b, labels, new_model_C(initial_readout))

# Sanity-check that modelC half of the hybrid model is as it was before
assert torch.allclose(hybrid_ac[1](xy_a), reference_model_C(xy_a))
assert torch.allclose(hybrid_bc[1](xy_b), reference_model_C(xy_b))

plt.plot(history_ac, label="stitched A")
plt.plot(history_bc, label="stitched B")
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("BCE loss")
plt.legend()
plt.show()

# Print learned stitching layer weights
print(" AC stitching layer weights:")
print(hybrid_ac[0].weight.detach().numpy())
print(" BC stitching layer weights:")
print(hybrid_bc[0].weight.detach().numpy())

acc_a = accuracy(xy_a, labels, hybrid_ac)
acc_b = accuracy(xy_b, labels, hybrid_bc)
print(f"Stitched AC accuracy: {acc_a:.3f}")
print(f"Stitched BC accuracy: {acc_b:.3f}")

# %% Visualize stitched models by (1) stitched features and (2) heatmap of hybrid predictions in original spaces

with torch.no_grad():
    xy_a_mapped = hybrid_ac[0](xy_a).numpy()
    xy_b_mapped = hybrid_bc[0](xy_b).numpy()

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(xy_a_mapped[:, 0], xy_a_mapped[:, 1], c=labels * 1.0)
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
ax[0].set_title("Stitched A reps")
ax[1].scatter(xy_b_mapped[:, 0], xy_b_mapped[:, 1], c=labels * 1.0)
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_title("Stitched B reps")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
model_predictions_heatmap(hybrid_ac, ax=ax[0])
ax[0].set_title(f"Hybrid AC output (Stitching)")
model_predictions_heatmap(hybrid_bc, ax=ax[1])
ax[1].set_title(f"Hybrid BC output (Stitching)")
fig.tight_layout()
plt.show()

# %% Do compatibility analysis; confirm that A beats B

history_ac2, hybrid_ac2 = continue_training_downstream(xy_a, labels, hybrid_ac)
history_bc2, hybrid_bc2 = continue_training_downstream(xy_b, labels, hybrid_bc)

# %% Prediction heatmaps of stitched + fine-tuned models

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
model_predictions_heatmap(hybrid_ac, ax=ax[0])
ax[0].set_title(f"Hybrid AC output (Fine-tuning)")
model_predictions_heatmap(hybrid_bc, ax=ax[1])
ax[1].set_title(f"Hybrid BC output (Fine-tuning)")
plt.show()

# %% Final plots and numbers

n_stitch_steps = len(history_ac)

fig = plt.figure(figsize=(3, 2))
plt.plot(history_ac + history_ac2, label="AC Hybrid")
plt.plot(history_bc + history_bc2, label="BC Hybrid")
plt.axvline(x=n_stitch_steps, ymin=0, ymax=1, color="k")
yl = plt.ylim()
plt.text(
    x=n_stitch_steps * 1.02,
    y=0.9 * max(yl) + 0.1 * min(yl),
    s=r"$\rightarrow$ Compatibility",
    horizontalalignment="left",
)
plt.text(
    x=n_stitch_steps * 0.95,
    y=0.9 * max(yl) + 0.1 * min(yl),
    s=r"Stitchability $\leftarrow$",
    horizontalalignment="right",
)
plt.xlabel("Training Step")
plt.ylabel("BCE loss")
plt.legend(loc="lower left")
fig.tight_layout()
plt.savefig("stitch_vs_compat.svg")
plt.show()

acc_a2 = accuracy(xy_a, labels, hybrid_ac2)
acc_b2 = accuracy(xy_b, labels, hybrid_bc2)
print(f"Fine-tuned AC accuracy: {acc_a2:.3f}")
print(f"Fine-tuned BC accuracy: {acc_b2:.3f}")