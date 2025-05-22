# %%
import pandas as pd
import mlflow
import numpy as np
from nn_lib.utils import search_runs_by_params
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm.auto import tqdm
import sys


mlflow_server_uri = "/data/projects/learnable-stitching/mlruns"

params = {
    "modelA": "resnet18",
    "modelB": "resnet34",
    # "layerA": "add_2",
    # "layerB": "add_4",
}

tstart = time.time()
all_runs = search_runs_by_params(
    experiment_name="learnable--stitching", params=params, tracking_uri=mlflow_server_uri
)
print("Finished search in ", time.time() - tstart, " seconds; found", len(all_runs), "runs")

# %%

mlflow_client = mlflow.MlflowClient(tracking_uri=mlflow_server_uri)


def get_metric_names(mlflow_client, run_id):
    run_data = mlflow_client.get_run(run_id).data.to_dictionary()
    metric_names = list(run_data["metrics"].keys())
    return metric_names


def pull_run_metrics_as_df(mlflow_client, run_id, metric_names=None, bin_steps: int = 1):
    if metric_names is None:
        metric_names = get_metric_names(mlflow_client, run_id)

    run: mlflow.ActiveRun = mlflow_client.get_run(run_id)
    info = run.data.params

    for_pd_collect = []
    for metric in metric_names:
        metric_history = mlflow_client.get_metric_history(run_id=run_id, key=metric)
        all_steps = [mm.step for mm in metric_history]
        all_ts = [mm.timestamp for mm in metric_history]
        all_vals = [mm.value for mm in metric_history]
        step_avg = [
            np.mean(all_steps[s : s + bin_steps]) for s in range(0, len(all_steps), bin_steps)
        ]
        ts_avg = [np.mean(all_ts[s : s + bin_steps]) for s in range(0, len(all_steps), bin_steps)]
        val_avg = [
            np.mean(all_vals[s : s + bin_steps]) for s in range(0, len(all_steps), bin_steps)
        ]
        pd_convertible_metric_history = [
            {
                "metric_name": metric,
                "step": s,
                "timestamp": t,
                "value": v,
                **info
            }
            for s, t, v in zip(step_avg, ts_avg, val_avg)
        ]
        for_pd_collect += pd_convertible_metric_history

    metrics_df = pd.DataFrame.from_records(for_pd_collect)
    return metrics_df


# %%

def main(option):

    if (option == "reuse"):
        big_df = pd.read_csv('big_df.csv')
    else:
        all_metrics = []
        for run in tqdm(all_runs.iterrows(), desc="Loading metrics", total=len(all_runs)):
            metrics_df = pull_run_metrics_as_df(mlflow_client, run[1].run_id, bin_steps=50)
            all_metrics.append(metrics_df)

        big_df = pd.concat(all_metrics)
        #big_df = big_df[(big_df["stitch_family"] == "sf")]
        big_df.to_csv('big_df.csv', index=False) 

    #The loss and validation of the donor models
    
    #print(big_df.to_string())
    
    df_pivot = big_df[["step", "layerA", "layerB", "metric_name", "value"]].pivot(index=["step", "layerA", "layerB"], columns="metric_name", values="value")
    df_pivot = df_pivot.reset_index()

    #print(df_pivot.to_string())
    donorA_loss = df_pivot.loc[0, "stitching-modelA-val-loss"]
    donorB_loss = df_pivot.loc[0, "stitching-modelB-val-loss"]
    
    donorA_val = 0.731846488499419
    donorB_val =  0.7877496706728785
    res50_val = 0.8876597536294036

    donor_val = {"resnet18": donorA_val, "resnet34": donorB_val, "resnet50": res50_val}
    '''print(donorA_val)'''



    # for metric, group in big_df.groupby("metric_name"):
    #df_stitching = big_df[big_df.metric_name == "stitching-modelAxB-val-loss"]
    df_training_curves = big_df[big_df.metric_name == "train_loss"]

    df_training_curves_sample = df_training_curves[(df_training_curves["layerA"] == "add_2") & (df_training_curves["layerB"] == "add_14") | #shallow to shallow
                                    (df_training_curves["layerA"] == "add_1") & (df_training_curves["layerB"] == "add_7")    | #shallow to mid
                                    (df_training_curves["layerA"] == "add_2") & (df_training_curves["layerB"] == "add_2")    | #shallow to deep
                                    (df_training_curves["layerA"] == "add_4") & (df_training_curves["layerB"] == "add_15")   | #mid to shallow
                                    (df_training_curves["layerA"] == "add_5") & (df_training_curves["layerB"] == "add_8")    | #mid to mid
                                    (df_training_curves["layerA"] == "add_5") & (df_training_curves["layerB"] == "add_3")    | #mid to deep
                                    (df_training_curves["layerA"] == "add_7") & (df_training_curves["layerB"] == "add_14")   | #deep to shallow
                                    (df_training_curves["layerA"] == "add_7") & (df_training_curves["layerB"] == "add_9")    | #deep to mid
                                    (df_training_curves["layerA"] == "add_6") & (df_training_curves["layerB"] == "add_4")      #deep to deep
                                    ]
    
    
    groupsA = {"add_1": "shallow", "add_2": "shallow", "add_4": "middle", "add_5": "middle", "add_7": "deep", "add_6": "deep"}
    groupsB = {"add_15": "shallow", "add_14": "shallow", "add_7": "middle", "add_8": "middle", "add_9": "middle", "add_2": "deep", "add_3": "deep", "add_4": "deep"}

    for i, row in df_training_curves_sample.iterrows():
        a = str(df_training_curves_sample.loc[i,"layerA"])
        b = str(df_training_curves_sample.loc[i,"layerB"])
        df_training_curves_sample.loc[i, "group"] = groupsA[a] + "-" + groupsB[b]
    
    df_validation = big_df[(big_df.metric_name == "stitching-modelAxB-val-loss") | (big_df.metric_name == "downstream-modelAxB-val-loss")]

    for i, row in df_validation.iterrows():
        df_validation.loc[i, "pairs"] = (df_validation.loc[i, "layerA"]) + "-" + (df_validation.loc[i,"layerB"])
        
        #adjusting the add to add_0 to make generating layer fractions easier
        if ((df_validation.loc[i, "layerA"]) == "add"):
            df_validation.loc[i, "layerA"] = "add_0"
        if ((df_validation.loc[i, "layerB"]) == "add"):
            df_validation.loc[i, "layerB"] = "add_0"

        #generating fractions for heatmap
        layerA = df_validation.loc[i, "layerA"]
        layerB = df_validation.loc[i, "layerB"]

        #find the under score, after those begin the index for the number of the layer, man probably should have used regex
        numLocA = layerA.find("_") + 1
        numLocB = layerB.find("_") + 1
        df_validation.loc[i, "layerAfrac"] = (float(layerA[numLocA:]) + 1) / (8.0 + 1) 
        df_validation.loc[i, "layerBfrac"] = (float(layerB[numLocB:]) + 1) / (16.0 + 1)

    #df_validation.info()
    #print(df_validation.to_string())


    df_validation_scatter = df_validation[["metric_name", "value", "pairs"]]
    df_validation_scatter = df_validation_scatter.pivot(index= "pairs", columns='metric_name', values='value')

    df_validation_diff = df_validation[["metric_name", "value", "pairs"]].pivot(index= "pairs", columns='metric_name', values='value')
    df_validation_diff["diff"] = df_validation_scatter["downstream-modelAxB-val-loss"] - df_validation_scatter["stitching-modelAxB-val-loss"]
    df_validation_diff["layerAfrac"] = df_validation["layerAfrac"]
    df_validation_diff["layerBfrac"] = df_validation["layerBfrac"]

    df_s_validation= df_validation[big_df.metric_name == "stitching-modelAxB-val-loss"]
    df_d_validation= df_validation[big_df.metric_name == "downstream-modelAxB-val-loss"]

    #df_s_validation[["layerA","layerAfrac", "layerB", "layerBfrac", "value"]].to_csv("debug.txt")

    #print(df_s_validation[["layerAfrac", "layerBfrac", "value"]].to_string)

    #graph of the different sample learning curves
    curves = plt.figure(figsize=(15, 6))
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=15)

    #plot the donor baselines
    plt.axhline(y=donorA_loss, color=(0,0,0), linestyle='--', label='Model A loss')
    plt.axhline(y=donorB_loss, color=(0,0,0), linestyle=':', label='Model B loss')

    #plot learning curves
    ax = sns.lineplot(x="step", y="value", data=df_training_curves_sample, hue="group")
    ax.set(xlabel='Batches', ylabel='Cross-Entropy Loss', 
           title= '(B) Sampled Learning Curves from Each Stitched Model Type (Including Downstream Learning)' +
                  ' Between ' + params["modelA"] + " and " + params["modelB"])

    sns.move_legend(ax, "upper right", title="", fontsize = 10)
    plt.axvline(11721.5, 0,10, color=(0,0,0)) #divider line for downstream learning
    plt.xlim(0.0, 20000)


    plt.tight_layout()

    plt.savefig("Sample_Learning_Curves_Graph.svg")
    plt.show()

    #graph for helping explore all the different learning cuvres, not for publishing 
    '''
    plt.figure()
    sns.lineplot(x="step", y="value", data=df_s_view, hue=df_s_view[["layerA", "layerB"]].apply(tuple, axis=1))
    sns.lineplot(x="step", y="value", data=df_d_view, hue=df_d_view[["layerA", "layerB"]].apply(tuple, axis=1))
    plt.savefig("Scout Graph")
    plt.show()'''

    #validation accuracy scatterpot for the validation before and after downstream learning
    
    plt.figure(figsize=(10,10))


    #plot baselines
    plt.axhline(y=donor_val[params["modelA"]], color=(0,0,0), linestyle='--', label='Model A Validation')
    plt.axhline(y=donor_val[params["modelB"]], color=(0,0,0), linestyle=':', label='Model B Validation')

    #plot useful dividing line
    #plot scatterplot of validation accuracies
    ax = sns.scatterplot(x="stitching-modelAxB-val-loss", y="downstream-modelAxB-val-loss", data=df_validation_scatter)
    plt.plot([0.0,1.0], [0.1,1.1], linestyle='-', color='grey', linewidth=2)
    
    plt.rc('font', size=15)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=25)

    ax.set(xlabel='Validation Accuracy Before Downstream Learning', ylabel='Validation Accuracy After Downstream Learning', )
    plt.title(label= '(C) Stitched Model Validation Accuracy ' + ' Between ' + params["modelA"] + " and " + params["modelB"], pad=20)
    plt.xlim(0.0, 0.8)
    plt.ylim(0.0, 0.8)

    plt.savefig("Validation_Graph.svg")
    plt.show()

    '''
    #heatmap subgraphs
    sns.set(font_scale=1.5)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(30,12))
    fig.suptitle('Validation Accuracy for All Possible Combinations of Stitched Models Between ' + params["modelA"] + " and " + params["modelB"])
    #Before downstream learning validation accuracy heatmap
    #plt.figure(figsize=(10, 7.5))
    sns.set(font_scale=1.5)
    df_hm = df_s_validation[["layerAfrac", "layerBfrac", "value"]].pivot(index="layerAfrac", columns="layerBfrac", values="value")
    ax1 = sns.heatmap(df_hm, ax=ax1)
    ax1.invert_yaxis()

    ax1.set(xlabel='Fractions of Components Donated by ModelB', ylabel='Fractions of Components Donated by ModelA', 
           title= "Before Downstream Learning") '''

    validation_pivot = df_validation[["layerA", "layerB", "metric_name", "value"]].pivot(
        index=["layerA", "layerB"], columns="metric_name", values="value"
    )

    validation_diff = (
        validation_pivot["downstream-modelAxB-val-loss"] - validation_pivot["stitching-modelAxB-val-loss"]
    )

    interesting_runs = validation_diff > 0.1

    # %% Diff as a heatmap, with rows = layerA, columns = layerB


    def layer_idx_getter(layer_name):
        try:
            return int(layer_name.split("_")[-1])
        except ValueError:
            return 0


    df_diff_matrix = validation_diff.unstack()
    df_diff_matrix = df_diff_matrix.reindex(
        sorted(df_diff_matrix.columns, key=layer_idx_getter), axis=1
    )

    max_diff = np.abs(df_diff_matrix.to_numpy()).flatten().max()

    plt.figure(figsize=(12.5, 12.5))
    #plt.rc('font', size=20)
    plt.rc('axes', titlesize=27.5)
    plt.rc('axes', labelsize=25)
    
    h = plt.imshow(
        df_diff_matrix.to_numpy(),
        cmap="PiYG",
        vmin=-max_diff,
        vmax=max_diff,
        extent=(0.0, 1.0, 0.0, 1.0),
    )
    plt.colorbar(shrink=0.6)
    plt.xticks(
        (np.arange(df_diff_matrix.shape[1]) + 0.5) / (df_diff_matrix.shape[1]),
        [
            f"{1-((1+l)/(df_diff_matrix.shape[1] + 1)):.1%}"
            for l in range(df_diff_matrix.shape[1])
        ],
        rotation=90,
    )
    plt.yticks(
        (np.arange(df_diff_matrix.shape[0]) + 0.5) / (df_diff_matrix.shape[0]),
        [
            f"{(1- ((1+l)/(df_diff_matrix.shape[0] + 1))):.1%}"
            for l in range(df_diff_matrix.shape[0])
        ],
    )

    plt.title(r"$\Delta$ Validation Accuracy")
    plt.xlabel("Number of Components Donated by " + params["modelB"])
    plt.ylabel("Number of Components Donated by " + params["modelA"])
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    #plt.tight_layout()


    plt.savefig("Diff_Heatmap.svg")
    plt.show()

    #heatmap for validation accuracy using matt plot lib

    df_s_matrix = df_s_validation[["layerA", "layerB", "value"]].pivot(index="layerA", columns="layerB", values= "value")
    df_s_matrix = df_s_matrix.reindex(
        sorted(df_s_matrix.columns, key=layer_idx_getter), axis=1
    )

    #print(df_s_matrix.to_string())

    ax = plt.figure(figsize=(12.5, 12.5))
    #plt.rc('font', size=20)
    plt.rc('axes', titlesize=27.5)
    plt.rc('axes', labelsize=25)
    
    h = plt.imshow(
        df_s_matrix.to_numpy(),
        cmap="magma",
        extent=(0.0, 1.0, 0.0, 1.0),

    )
    plt.colorbar(shrink=0.6)
    plt.xticks(
        (np.arange(df_s_matrix.shape[1]) + 0.5) / (df_s_matrix.shape[1]),
        [
            f"{(1 - ((1+l)/(df_s_matrix.shape[1] + 1))):.1%}"
            for l in range(df_s_matrix.shape[1])
        ],
        rotation=90, fontsize = 20
    )
    plt.yticks(
        (np.arange(df_s_matrix.shape[0]) + 0.5) / (df_s_matrix.shape[0]),
        [
            f"{(1 - (1+l)/(df_s_matrix.shape[0] + 1)):.1%}"
            for l in range(df_s_matrix.shape[0])
        ], fontsize = 20
    )
    
    plt.title(r"Validation Accuracy Without Downstream Learning")
    plt.xlabel("Number of Components Donated by " + params["modelB"])
    plt.ylabel("Number of Components Donated by " + params["modelA"])
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    #plt.tight_layout()


    plt.savefig("Val_Heatmap.svg")
    plt.show()

if __name__ == "__main__":
    args = sys.argv[1:]
    option = args[0]
    main(option)