# %%
from math import sqrt

import mlflow
import torch
import os
from dotenv import load_dotenv
from nn_lib.utils import load_artifact, RunningVariance
from tqdm.auto import tqdm

from experiment import DonorSpec, create_hybrid_model

##=================================================================================##
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    load_dotenv() 

    user = os.getenv("MLFLOW_SQL_USERNAME")
    password = os.getenv("MLFLOW_SQL_PASSWORD")
    db_name = os.getenv("MLFLOW_SQL_DATABASE")

    mlflow.set_tracking_uri(f"mysql+pymysql://{user}:{password}@localhost:3306/{db_name}")
    mlflow.set_experiment("learnable-stitching-v0.5")

    if args.run_id is None:
        runs = mlflow.search_runs(output_format="list")
        print("Available runs:")
        for run in runs:
            print(run.info.run_id)
        exit(0)


    client = mlflow.MlflowClient()
    og_run = client.get_run(args.run_id)
    if "finetuning-posthoc-val-acc-mcse" in og_run.data.metrics:
        print("already computed")
        exit(0)

    delta_dicts: dict = {
        "stitching": load_artifact("weights/stitching-modelAxB.pt", args.run_id),
        "finetuning": load_artifact("weights/finetuning-modelAxB.pt", args.run_id),
    }

    for phase, deltas in delta_dicts.items():
        delta_dict: dict = deltas
        params = og_run.data.params
        device = params["device"]

        #print(args.run_id)
        #exit(0)

        # print(params)
        donorA = DonorSpec(params["donorA.model"], params["donorA.layer"], params["donorA.dataset"])
        donorB = DonorSpec(params["donorB.model"], params["donorB.layer"], params["donorB.dataset"])
        hybrid_model, _, _ = create_hybrid_model(
            donorA=donorA, donorB=donorB, stitch_family=params["stitch_family"], plot_architecture=False
        )
        hybrid_model.load_delta_state_dict(delta_dict, base_model=hybrid_model, strict=True)
        hybrid_model = hybrid_model.eval().to(device)

        #donorB.dataset.prepare_data()
        donorB.dataset.setup("val")
        val_data = donorB.dataset.val_dataloader(batch_size=50, num_workers=4, pin_memory=True)

        loss_stats = RunningVariance()
        acc_stats = RunningVariance()

        with torch.no_grad():
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=255)
            for im, la in tqdm(val_data):
                im, la = im.to(device), la.to(device)
                out = hybrid_model(im)
                is_corrects = (torch.argmax(out, dim=1) == la).flatten(start_dim=1, end_dim=2).float()
                losses = loss_fn(out, la)

                #print(out.shape, la.shape, is_corrects.shape)
                loss_stats.update(losses)
                acc_stats.update(is_corrects)
                #print(loss_stats.mu_x.shape)

        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-loss-mean",
            value=loss_stats.mu_x.mean(),
        )
        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-loss-mcse",
            value=torch.max(torch.sqrt(loss_stats.variance / len(val_data.dataset))),
        )
        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-acc-mean",
            value=acc_stats.mu_x.mean(),
        )
        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-acc-mcse",
            value=torch.max(torch.sqrt(acc_stats.variance / len(val_data.dataset))),
        )
