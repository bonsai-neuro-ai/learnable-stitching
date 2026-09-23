# Python script, a quick fix to replace experiment runs by filtering the dataframe
import json
import subprocess
from nn_lib.utils import save_as_artifact, search_runs_by_params
import mlflow
import time

mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
mlflow.set_experiment("learnable-stitching-v0.4")
client = mlflow.MlflowClient()

params = {
    "target_type": "TargetType.MATCH_DOWNSTREAM",
}

tstart = time.time()
print("Begin Search")
df = search_runs_by_params(
    experiment_name="learnable-stitching-v0.4", params=params, finished_only=True
)
print("Finished search in ", time.time() - tstart)

filtered_df = df[(df["metrics.stitching layer converged"] == 1) & (df["metrics.steps to train stitching layer"] == 10000)]
filtered_df = filtered_df.reset_index(drop=True)
filtered_df = filtered_df["run_id"]

for i in range(filtered_df.shape[0]):
    run = client.get_run(filtered_df.at[i])
    run_par = run.data.params

    target_type = ((params["target_type"].split("."))[-1])
    args = ["python", "experiment.py",
                     "--stitch_family", run_par["stitch_family"], "--target_type", target_type, "--init_batches", run_par["init_batches"], "--batch_size", run_par["batch_size"], 
                     "--donorA.model", run_par["donorA_model"], "--donorA.layer", run_par["donorA_layer"], "--donorA.dataset", run_par["donorA_dataset"],
                     "--donorB.model", run_par["donorB_model"], "--donorB.layer", run_par["donorB_layer"], "--donorB.dataset", run_par["donorB_dataset"]
                    ]
    
    print(filtered_df)
    client.delete_run(filtered_df.at[i])
    subprocess.run(args)

    