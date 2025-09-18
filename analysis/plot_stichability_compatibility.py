# %%
import time
from functools import partial

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib import colormaps as cm
from nn_lib.utils import search_runs_by_params

from analysis.utils import pivot_heatmap, layer_idx, get_metric_history

mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
mlflow.set_experiment("learnable-stitching-v0.4")
client = mlflow.MlflowClient()

params = {
    "target_type": "TargetType.TASK",
}

tstart = time.time()
df = search_runs_by_params(
    experiment_name="learnable-stitching-v0.4", params=params, finished_only=True
)
print("Finished search in ", time.time() - tstart)

all_params = [c for c in df.columns if c.startswith("params.")]
all_metrics = [c for c in df.columns if c.startswith("metrics.")]


def shortname(row, ab):
    model = row[f"params.donor{ab}_model"]
    layer = row[f"params.donor{ab}_layer"]
    return model[0] + model[-2:] + f"_{layer_idx(layer):02d}"


df["upstream"] = df.apply(partial(shortname, ab="A"), axis=1)
df["downstream"] = df.apply(partial(shortname, ab="B"), axis=1)

# %% Data checks

if (set(df["upstream"]) - set(df["downstream"])) or (set(df["downstream"]) - set(df["upstream"])):
    print("Warning: some missing runs; upstream and downstream sets differ")


groups = df.groupby(["params.donorA_model", "params.donorB_model"])
for models, group in groups:
    print(models, len(group))

# Check for any duplicates or missing combinations of layers x models
dup = False
for models, group in groups:
    layersA = group["params.donorA_layer"].unique()
    layersB = group["params.donorB_layer"].unique()
    expected_combinations = set((a, b) for a in layersA for b in layersB)
    actual_combinations = set(
        (row["params.donorA_layer"], row["params.donorB_layer"]) for _, row in group.iterrows()
    )
    missing_combinations = expected_combinations - actual_combinations
    if missing_combinations:
        print(f"Missing combinations for models {models}: {missing_combinations}")
    duplicates = group.duplicated(subset=["params.donorA_layer", "params.donorB_layer"], keep=False)
    if duplicates.any():
        print(f"Duplicate entries found for models {models}:")
        print(*[name[7:] for name in all_params], sep="\t")
        for idx, row in group[duplicates].iterrows():
            print(*row[all_params], sep="\t")
        dup = True
if not dup:
    print("No duplicate rows found; parameter combinations look good.")

# %% Plot everything everywhere all at once

plt.figure(figsize=(10, 10))
pivot_heatmap(
    df,
    metric="metrics.stitching-modelAxB-val-loss",
    row_col="upstream",
    col_col="downstream",
    vmin=0,
    vmax=10,
)
plt.title("Stitching loss")
plt.xlabel("Downstream model")
plt.ylabel("Upstream model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
pivot_heatmap(
    df,
    metric="metrics.downstream-modelAxB-val-loss",
    row_col="upstream",
    col_col="downstream",
    vmin=0,
    vmax=10,
)
plt.title("Fine-tuning loss")
plt.xlabel("Downstream model")
plt.ylabel("Upstream model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% Plotting by model

# for (modelA, modelB), group in groups:
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#
#     print(f"Plotting heatmaps for {modelA} x {modelB}")
#     pivot_heatmap(
#         group,
#         metric="metrics.stitching-modelAxB-val-loss",
#         row_col="params.donorA_layer",
#         col_col="params.donorB_layer",
#         vmin=0,
#         vmax=10,
#         ax=ax[0],
#         sort_key=layer_idx,
#     )
#     ax[0].set_xlabel(modelB)
#     ax[0].set_ylabel(modelA)
#     ax[0].set_title("Stitching loss")
#
#     pivot_heatmap(
#         group,
#         metric="metrics.downstream-modelAxB-val-loss",
#         row_col="params.donorA_layer",
#         col_col="params.donorB_layer",
#         vmin=0,
#         vmax=10,
#         ax=ax[1],
#         sort_key=layer_idx,
#     )
#     ax[1].set_xlabel(modelB)
#     ax[1].set_ylabel(modelA)
#     ax[1].set_title("Fine-tuning loss")
#     fig.tight_layout()
#     plt.show()

# %% Plots for loss curves


def color_for_model_layer(shortname):
    model, layer = shortname.split("_")
    color_group = cm["tab20"].colors
    color_idx = ["r18", "r34", "r50", "r101"].index(model)
    color1, color2 = color_group[color_idx * 2], color_group[color_idx * 2 + 1]
    frac = int(layer) / 15
    color = tuple(c1 * (1 - frac) + c2 * frac for c1, c2 in zip(color1, color2))
    return color


def plot_loss_sequence(run_id, *, color=None, label=None, bin_steps=10, ax=None):
    ax = ax or plt.gca()

    hist_stitch = get_metric_history(
        run_id,
        "stitching-modelAxB-train-loss",
        mlflow_client=client,
        bin_steps=bin_steps,
    )
    hist_finetune = get_metric_history(
        run_id,
        "downstream-modelAxB-train-loss",
        mlflow_client=client,
        bin_steps=bin_steps,
    )

    # stitching results are staggered, some very short and others long runs. Reset the 'time'
    # axis to negative values so that time=0 is always the end of stitching and start of
    # finetuning
    hist_stitch["neg_step"] = hist_stitch["step"] - hist_stitch["step"].max()

    # Often finetuning accomplishes a lot in the first few steps, which is hidden by the binning
    # done by 'bin_steps'. To *visually* fix this, prepend a copy of the last stitching row with
    # step=0, since performance at end of stitching is the starting point for finetuning.
    copy_of_last_stitching_row = hist_stitch.iloc[-1].to_dict()
    copy_of_last_stitching_row["step"] = 0
    hist_finetune = pd.concat([pd.DataFrame([copy_of_last_stitching_row]), hist_finetune])

    ax.plot(
        hist_stitch["neg_step"],
        hist_stitch["value"],
        color=color,
    )
    ax.plot(
        hist_finetune["step"],
        hist_finetune["value"],
        label=label,
        color=color,
    )


for downstream, group in df.groupby("downstream"):
    if downstream != "r50_06":
        continue

    plt.figure()
    for _, row in group.iterrows():
        upstream = row["upstream"]
        plot_loss_sequence(
            row["run_id"],
            color=color_for_model_layer(upstream),
            label=f"{upstream}" if any(upstream.endswith(s) for s in ["00", "07", "15"]) else None,
            bin_steps=10,
        )

    plt.yscale("log")
    plt.xlim(-1005, 1005)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(title="Upstream")
    plt.savefig(f"analysis/plots/stitch_compat_loss_curves_{downstream}.svg")
    plt.show()


# %% Check for rank-order violations and print out their loss info

metric1 = "metrics.stitching-modelAxB-train-loss"
metric2 = "metrics.downstream-modelAxB-train-loss"
for key, grp in df.groupby("downstream"):
    winner1 = grp.loc[grp[metric1].idxmin()]
    winner2 = grp.loc[grp[metric2].idxmin()]
    loss1_on_1 = winner1[metric1]  # Performance of best-stitching layer after stitching
    loss2_on_2 = winner2[metric2]  # Performance of best-finetuning layer after finetuning
    loss1_on_2 = winner1[metric2]  # Performance of best-stitching layer after finetuning
    loss2_on_1 = winner2[metric1]  # Performance of best-finetuning layer after stitching

    is_rank_order_violation = winner1["upstream"] != winner2["upstream"]
    is_sane = loss1_on_2 < loss1_on_1 and loss2_on_2 < loss2_on_1
    print(
        key,
        is_rank_order_violation,
        is_sane,
        winner1["upstream"],
        winner2["upstream"],
        loss1_on_1,
        loss2_on_1,
        loss1_on_2,
        loss2_on_2,
        sep="\t",
    )

    if is_rank_order_violation:
        plt.figure(figsize=(4, 3))
        plot_loss_sequence(
            winner1["run_id"],
            color="C0",
            label=f"Best stitching: {winner1['upstream']}",
            bin_steps=1,
        )
        plot_loss_sequence(
            winner2["run_id"],
            color="C1",
            label=f"Best finetuning: {winner2['upstream']}",
            bin_steps=1,
        )
        plt.yscale("log")
        plt.xlim(-1005, 1005)
        plt.xlabel("\n".join(["Step", r"Stitching $\leftarrow$  $\rightarrow$ Finetuning"]))
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Rank-order violation on downstream {key}")
        plt.tight_layout()
        plt.savefig(f"analysis/plots/rank_order_violation_loss_curves_{key}.svg")
        plt.show()
