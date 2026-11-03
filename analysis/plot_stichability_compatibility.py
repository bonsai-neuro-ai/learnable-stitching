# %%
import time
from csv import DictWriter
from functools import partial

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps as cm
from nn_lib.utils import search_runs_by_params
from scipy.stats import norm

from analysis.utils import pivot_heatmap, layer_idx, get_metric_history

mlflow.set_tracking_uri("sqlite:////data/projects/learnable-stitching/mlflow.db")
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

df = df[df["params.stitch_init"].isna()]

# %%

all_params = [c for c in df.columns if c.startswith("params.")]
all_metrics = [c for c in df.columns if c.startswith("metrics.")]


def shortname(row, ab):
    model = row[f"params.donor{ab}_model"]
    layer = row[f"params.donor{ab}_layer"]
    return model[0] + model[-2:] + f"_{layer_idx(layer):02d}"


def shortname_marked(row, ab, mark):
    model = row[f"params.donor{ab}_model"] + mark
    layer = row[f"params.donor{ab}_layer"]
    return model[0] + model[-2:] + f"_{layer_idx(layer):02d}"


df["upstream"] = df.apply(partial(shortname, ab="A"), axis=1)
df["downstream"] = df.apply(partial(shortname, ab="B"), axis=1)

# %% Data checks

if (set(df["upstream"]) - set(df["downstream"])) or (set(df["downstream"]) - set(df["upstream"])):
    print("Warning: some missing runs; upstream and downstream sets differ")


groups = df.groupby(["params.donorA_model", "params.donorB_model"])

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
            for other_idx, other_row in group[duplicates].iterrows():
                if idx == other_idx:
                    continue
                if (row["params.donorA_layer"], row["params.donorB_layer"]) == (
                    other_row["params.donorA_layer"],
                    other_row["params.donorB_layer"],
                ):
                    print("=>", *other_row[all_params], sep="\t")
        dup = True
    break
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
plt.xlabel("Downstream model", fontsize=10)
plt.ylabel("Upstream model", fontsize=10)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# plt.savefig(f"analysis/plots/stitching_modelAxB_val_loss.svg")

plt.figure(figsize=(10, 15))
pivot_heatmap(
    df,
    metric="metrics.downstream-modelAxB-val-loss",
    row_col="upstream",
    col_col="downstream",
    vmin=0,
    vmax=10,
)
plt.title("Fine-tuning loss")
plt.xlabel("Downstream model", fontsize=10)
plt.ylabel("Upstream model", fontsize=10)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# plt.savefig(f"analysis/plots/downstream_modelAxB_val_loss.svg")

# %% Plotting by model

for (modelA, modelB), group in groups:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    group["params.donorA_layer"] = group["params.donorA_layer"].map(lambda a: "up-" + a)

    cell_width = 0.1
    cell_height = 0.1

    print(f"Plotting heatmaps for {modelA} x {modelB}")
    pivot_heatmap(
        group,
        metric="metrics.stitching-modelAxB-val-loss",
        row_col="params.donorA_layer",
        col_col="params.donorB_layer",
        vmin=0,
        vmax=10,
        ax=ax[0],
        sort_key=layer_idx,
    )
    x_labels = [
        (
            layer.get_text()
            if len(layer.get_text().split("_")) > 1
            and (int(layer.get_text().split("_")[1]) + 1) % 2 == 0
            else ""
        )
        for layer in ax[0].get_xticklabels()
    ]
    y_labels = [
        (
            layer.get_text()
            if len(layer.get_text().split("_")) > 1
            and (int(layer.get_text().split("_")[1]) + 1) % 2 == 0
            else ""
        )
        for layer in ax[0].get_yticklabels()
    ]

    ax[0].set_xlabel(modelB, fontsize=20)
    ax[0].set_ylabel(modelA, fontsize=20)
    ax[0].set_title("Stitching loss", fontsize=24)
    ax[0].set_xticklabels(x_labels, fontsize=12)
    ax[0].set_yticklabels(y_labels, fontsize=12, rotation=0)
    # plt.savefig(f"analysis/plots/stitching_{modelA}x{modelB}_val_loss.svg")

    pivot_heatmap(
        group,
        metric="metrics.downstream-modelAxB-val-loss",
        row_col="params.donorA_layer",
        col_col="params.donorB_layer",
        vmin=0,
        vmax=10,
        ax=ax[1],
        sort_key=layer_idx,
    )
    x_labels = [
        (
            layer.get_text()
            if len(layer.get_text().split("_")) > 1
            and (int(layer.get_text().split("_")[1]) + 1) % 2 == 0
            else ""
        )
        for layer in ax[1].get_xticklabels()
    ]
    y_labels = [
        (
            layer.get_text()
            if len(layer.get_text().split("_")) > 1
            and (int(layer.get_text().split("_")[1]) + 1) % 2 == 0
            else ""
        )
        for layer in ax[1].get_yticklabels()
    ]

    ax[1].set_xlabel(modelB, fontsize=20)
    ax[1].set_ylabel(modelA, fontsize=20)
    ax[1].set_title("Fine-tuning loss", fontsize=24)
    ax[1].set_xticklabels(x_labels, fontsize=12)
    ax[1].set_yticklabels(y_labels, fontsize=12, rotation=0)
    fig.tight_layout()
    plt.show()
    plt.savefig(f"analysis/plots/{modelA}x{modelB}_val_loss.svg")


# %% Plots for loss curves
def color_for_model_layer(shortname):
    model, layer = shortname.split("_")
    color_group = cm["tab20"].colors
    color_idx = ["r18", "r34", "r50", "r101"].index(model)
    color1, color2 = color_group[color_idx * 2], color_group[color_idx * 2 + 1]
    frac = int(layer) / 15
    color = tuple(c1 * (1 - frac) + c2 * frac for c1, c2 in zip(color1, color2))
    return color


def lognormal_quantiles(mean, std, ps):
    """
    Estimate quantiles of a lognormal distribution from
    the mean and std in linear space.

    Parameters
    ----------
    mean : float
        Mean of X (in linear space)
    std : float
        Standard deviation of X (in linear space)
    ps : array-like
        Probabilities (between 0 and 1) for which quantiles are requested
    Returns
    -------
    quantiles : np.ndarray
        Quantile estimates at probabilities ps
    """
    mean = float(mean)
    std = float(std)
    ps = np.asarray(ps)

    if mean <= 0:
        raise ValueError("Mean must be > 0 for lognormal assumption.")
    if std <= 0:
        raise ValueError("Std must be > 0.")

    var = std**2
    sigma2 = np.log(1 + var / mean**2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2

    return np.exp(mu + sigma * norm.ppf(ps))


def lognormal_quantiles_errorbars(x, y, yerr, quant=0.16, input_type=None, **kwargs):
    """
    Plot y vs x with asymmetric lognormal error bars in y.

    Parameters
    ----------
    x : array-like
        x-coordinates of data points
    y : array-like
        y-coordinates of data points (mean values)
    yerr : array-like
        Standard deviation of y-coordinates (in log space)
    **kwargs : dict
        Additional keyword arguments passed to plt.errorbar

    Returns
    -------
    lines : list
        List of Line2D objects created by plt.errorbar
    """

    y = np.asarray(y)
    yerr = np.asarray(yerr)

    if np.any(y <= 0):
        raise ValueError("All y values must be > 0 for lognormal error bars.")
    if np.any(yerr <= 0):
        raise ValueError("All yerr values must be > 0 for lognormal error bars.")

    # Get asymmetric error bars from lognormal quantiles
    if input_type == "linear":
        lower, upper = zip(
            *[lognormal_quantiles(y_, yerr_, [quant, 1 - quant]) for y_, yerr_ in zip(y, yerr)]
        )
    else:
        lower, upper = zip(
            *[y_ + np.sqrt(yerr_) * norm.ppf([quant, 1 - quant]) for y_, yerr_ in zip(y, yerr)]
        )

    lower_err = y - np.asarray(lower)
    upper_err = np.asarray(upper) - y
    asymmetric_err = np.vstack((lower_err, upper_err))

    return plt.errorbar(x, y, yerr=asymmetric_err, fmt="o", ms=0.5, **kwargs)


def plot_loss_sequence(
    run_id,
    *,
    stitch_mean=None,
    tune_mean=None,
    stitch_error=None,
    tune_error=None,
    color=None,
    label=None,
    bin_steps=10,
    ax=None,
):
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
    rev_max = hist_stitch["step"].max()
    # hist_stitch["neg_step"] = hist_stitch["neg_step"].dropna()
    # hist_finetune["step"] = hist_finetune["stepas ups"].dropna()
    # Often finetuning accomplishes a lot in the first few steps, which is hidden by the binning
    # done by 'bin_steps'. To *visually* fix this, prepend a copy of the last stitching row with
    # step=0, since performance at end of stitching is the starting point for finetuning.
    copy_of_last_stitching_row = hist_stitch.iloc[-1].to_dict()
    copy_of_last_stitching_row["step"] = 0
    hist_finetune = pd.concat([pd.DataFrame([copy_of_last_stitching_row]), hist_finetune])

    # stitch_spline = make_interp_spline(hist_stitch["neg_step"], hist_stitch["value"])
    # if (list(hist_finetune["step"])[-1] == list(hist_finetune["step"])[-2]):
    # print(run_id)

    X_stitch = hist_stitch["neg_step"]
    Y_stitch = hist_stitch["value"]
    # X_stitch = np.linspace(hist_stitch["neg_step"].min(), hist_stitch["neg_step"].max(), 1000)
    # Y_stitch = stitch_spline(X_stitch)

    # finetune_spline = make_interp_spline(hist_finetune["step"], hist_finetune["value"])
    # X_finetune = np.linspace(hist_finetune["step"].min(), hist_finetune["step"].max(), 100)
    # Y_finetune = finetune_spline(X_finetune)
    X_finetune = hist_finetune["step"]
    Y_finetune = hist_finetune["value"]

    window_size = 25
    # Create normalized weights for SMA
    weights = np.ones(window_size) / window_size

    # Calculate the simple moving average
    if len(hist_stitch["step"]) > window_size:
        Y_st_avg = np.convolve(Y_stitch, weights, mode="valid")

        # print(Y_st_avg)
        x_st_avg = np.arange(0, len(hist_stitch["step"]) - window_size + 1, 1)
        x_st_neg = x_st_avg - x_st_avg.max()

        ax.plot(
            x_st_neg,
            Y_st_avg,
            color=color,
        )

        # x_labels = [step + rev_max if str(step.get_text()).isdigit() and int(str(step.get_text()).replace('−', '-')) < 0 and step in hist_stitch["neg_step"]
        # else step + rev_max if str(step.get_text()).isdigit() and int(str(step.get_text()).replace('−', '-')) > 0 and step in hist_finetune["step"]
        # else ""
        # for step in  ax.get_xticklabels()]

    Y_tu_avg = np.concatenate(
        (np.array([list(hist_stitch["value"])[-1]]), np.convolve(Y_finetune, weights, mode="valid"))
    )

    x_tu_avg = np.arange(0, len(hist_finetune["step"]) - window_size + 2, 1)

    ax.plot(X_stitch, Y_stitch, color=color, alpha=0.33)

    ax.plot(X_finetune, Y_finetune, label=label, color=color, alpha=0.33)
    ax.plot(
        x_tu_avg,
        Y_tu_avg,
        color=color,
    )

    # ax.set_xticklabels(x_labels)

    if stitch_mean != None and stitch_error != None:
        x_point = 0
        y_point = [stitch_mean]
        lognormal_quantiles_errorbars(
            x_point,
            y_point,
            yerr=[stitch_error],
            input_type="linear",
            color=color,
            ecolor=color,
            capsize=3,
            label=(label.split(" ")[-1] + " stitching validation mean"),
        )

    if tune_mean != None and tune_error != None:
        x_point = [list(hist_finetune["step"])[-1] + 25]
        y_point = [tune_mean]
        lognormal_quantiles_errorbars(
            x_point,
            y_point,
            yerr=[tune_error],
            input_type="linear",
            color=color,
            ecolor=color,
            capsize=3,
            label=(label.split(" ")[-1] + " fine-tuning validation mean"),
        )


# for downstream, group in df.groupby("downstream"):
# plt.figure()
# for _, row in group.iterrows():
#     upstream = row["upstream"]
#     plot_loss_sequence(
#         row["run_id"],
#         color=color_for_model_layer(upstream),
#         label=f"{upstream}" if any(upstream.endswith(s) for s in ["00", "07", "15"]) else None,
#         bin_steps=10,
#     )
#
# plt.yscale("log")
# plt.xlim(-1505, 1005)
# plt.xlabel("Step")
# plt.ylabel("Loss")
# plt.legend(title="Upstream")
# plt.title(downstream)
# plt.savefig(f"analysis/plots/stitch_compat_loss_curves_{downstream}.svg")
# plt.show()

# %% sanity-check the posthoc error values

plt.figure(figsize=(3, 3))
sns.scatterplot(
    df, x="metrics.stitching-modelAxB-val-loss", y="metrics.stitching-posthoc-val-loss-mean"
)
sns.scatterplot(
    df, x="metrics.downstream-modelAxB-val-loss", y="metrics.downstream-posthoc-val-loss-mean"
)
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(3, 3))
sns.scatterplot(
    df, x="metrics.stitching-modelAxB-val-acc1", y="metrics.stitching-posthoc-val-acc-mean"
)
sns.scatterplot(
    df, x="metrics.downstream-modelAxB-val-acc1", y="metrics.downstream-posthoc-val-acc-mean"
)
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Print big table of information to a file


def a_bigger_than_b_test(mean_a, mcse_a, mean_b, mcse_b):
    """If $|z| > 1.96$: significant at α = 0.05 (roughly 95% confidence)
    If $|z| > 2.58$: significant at α = 0.01 (roughly 99% confidence)
    If $|z| > 1$: probably worth noting but not formally significant
    """
    z = (mean_a - mean_b) / np.sqrt(mcse_a * mcse_a + mcse_b * mcse_b)
    return z


LOSS_AFTER_STITCHING = "metrics.stitching-modelAxB-val-loss"
LOSS_AFTER_FINETUNING = "metrics.downstream-modelAxB-val-loss"
ACC_AFTER_STITCHING = "metrics.stitching-modelAxB-val-acc1"
ACC_AFTER_FINETUNING = "metrics.downstream-modelAxB-val-acc1"
LOSS_AFTER_STITCHING_MCSE = "metrics.stitching-posthoc-val-loss-mcse"
LOSS_AFTER_FINETUNING_MCSE = "metrics.downstream-posthoc-val-loss-mcse"
ACC_AFTER_STITCHING_MCSE = "metrics.stitching-posthoc-val-acc-mcse"
ACC_AFTER_FINETUNING_MCSE = "metrics.downstream-posthoc-val-acc-mcse"

perf_mean_var = pd.read_csv("analysis/mv.csv")
mv_stitch = perf_mean_var[perf_mean_var["phase"] == "stitching"].set_index("run_id")
mv_tune = perf_mean_var[perf_mean_var["phase"] == "downstream"].set_index("run_id")

fig_scatter = plt.figure(figsize=(5, 5))

with open("analysis/summary_table.csv", "w") as file:
    writer = DictWriter(
        file,
        fieldnames=[
            "Downstream",
            "Best Stitching",
            "BS Stitching Loss",
            "BS Stitching Accuracy",
            "BS Fine-Tuning Loss",
            "BS Fine-Tuning Accuracy",
            "Best Fine-Tuning",
            "BFT Stitching Loss",
            "BFT Stitching Accuracy",
            "BFT Fine-Tuning Loss",
            "BFT Fine-Tuning Accuracy",
            "test_statistic_stitching",
            "test_statistic_finetuning",
            "is_rank_order_violation",
            "winner1_runid",
            "winner2_runid",
        ],
    )
    writer.writeheader()

    for key, grp in df.groupby("downstream"):
        winner1 = grp.loc[grp[LOSS_AFTER_STITCHING].idxmin()]
        winner2 = grp.loc[grp[LOSS_AFTER_FINETUNING].idxmin()]

        # check convergence
        converged = (
            winner1["metrics.stitching layer converged"]
            and winner2["metrics.stitching layer converged"]
        )

        is_rov = winner1["upstream"] != winner2["upstream"]
        test_statistic_stitching = a_bigger_than_b_test(
            winner1[ACC_AFTER_STITCHING],
            winner1[ACC_AFTER_STITCHING_MCSE],
            winner2[ACC_AFTER_STITCHING],
            winner2[ACC_AFTER_STITCHING_MCSE],
        )
        test_statistic_finetuning = a_bigger_than_b_test(
            winner2[ACC_AFTER_FINETUNING],
            winner2[ACC_AFTER_FINETUNING_MCSE],
            winner1[ACC_AFTER_FINETUNING],
            winner1[ACC_AFTER_FINETUNING_MCSE],
        )

        row = {
            "Downstream": key,
            "Best Stitching": winner1["upstream"],
            "BS Stitching Loss": f"{winner1[LOSS_AFTER_STITCHING]:.3f}±{winner1[LOSS_AFTER_STITCHING_MCSE]:.3f}",
            "BS Stitching Accuracy": f"{winner1[ACC_AFTER_STITCHING]:.2%}±{winner1[ACC_AFTER_STITCHING_MCSE]:.2%}",
            "BS Fine-Tuning Loss": f"{winner1[LOSS_AFTER_FINETUNING]:.3f}±{winner1[LOSS_AFTER_FINETUNING_MCSE]:.3f}",
            "BS Fine-Tuning Accuracy": f"{winner1[ACC_AFTER_FINETUNING]:.2%}±{winner1[ACC_AFTER_FINETUNING_MCSE]:.2%}",
            "Best Fine-Tuning": winner2["upstream"],
            "BFT Stitching Loss": f"{winner2[LOSS_AFTER_STITCHING]:.3f}±{winner2[LOSS_AFTER_STITCHING_MCSE]:.3f}",
            "BFT Stitching Accuracy": f"{winner2[ACC_AFTER_STITCHING]:.2%}±{winner2[ACC_AFTER_STITCHING_MCSE]:.2%}",
            "BFT Fine-Tuning Loss": f"{winner2[LOSS_AFTER_FINETUNING]:.3f}±{winner2[LOSS_AFTER_FINETUNING_MCSE]:.3f}",
            "BFT Fine-Tuning Accuracy": f"{winner2[ACC_AFTER_FINETUNING]:.2%}±{winner2[ACC_AFTER_FINETUNING_MCSE]:.2%}",
            "is_rank_order_violation": is_rov,
            "test_statistic_stitching": test_statistic_stitching,
            "test_statistic_finetuning": test_statistic_finetuning,
            "winner1_runid": winner1["run_id"],
            "winner2_runid": winner2["run_id"],
        }
        writer.writerow(row)

        # Check for WTFs
        if (winner1[ACC_AFTER_STITCHING] > winner1[ACC_AFTER_FINETUNING]) or (winner2[ACC_AFTER_STITCHING] > winner2[ACC_AFTER_FINETUNING]):
            print("WTF", key)

        if (test_statistic_stitching > 2.58 and test_statistic_finetuning > 2.58):
            perf_diff = winner2[ACC_AFTER_FINETUNING] - winner1[ACC_AFTER_FINETUNING]
            perf_diff_err = np.sqrt(winner2[ACC_AFTER_FINETUNING_MCSE]**2 + winner1[ACC_AFTER_FINETUNING_MCSE]**2)
            print(key, "∆", perf_diff, "±", perf_diff_err)

        # for all ROVs, plot accuracy vs accuracy
        if winner1["upstream"] != winner2["upstream"]:
            plt.errorbar(
                x=[winner1[ACC_AFTER_STITCHING], winner1[ACC_AFTER_FINETUNING]],
                xerr=[
                    3 * winner1[ACC_AFTER_STITCHING_MCSE],
                    3 * winner1[ACC_AFTER_FINETUNING_MCSE],
                ],
                y=[winner2[ACC_AFTER_STITCHING], winner2[ACC_AFTER_FINETUNING]],
                yerr=[
                    3 * winner2[ACC_AFTER_STITCHING_MCSE],
                    3 * winner2[ACC_AFTER_FINETUNING_MCSE],
                ],
                label=key,
                marker="o" if (test_statistic_stitching > 2.58 and test_statistic_finetuning > 2.58) else "."
            )
    plt.legend()
    plt.xlabel("Best-Stitched Model Accuracy S--FT")
    plt.ylabel("Best-Fine-Tuned Model Accuracy S--FT")
    plt.plot(*plt.xlim(), *plt.xlim(), "--", color="gray")
    plt.axis("equal")
    plt.show()

print("done")
