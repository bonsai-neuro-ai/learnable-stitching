# %%
from math import sqrt

import mlflow
import torch
from nn_lib.utils import load_artifact, RunningVariance
from tqdm.auto import tqdm

from experiment import DonorSpec, create_hybrid_model

##=================================================================================##
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:////data/projects/learnable-stitching/mlflow.db")
    mlflow.set_experiment("learnable-stitching-v0.4")

    if args.run_id is None:
        runs = mlflow.search_runs(output_format="list")
        print("Available runs:")
        for run in runs:
            print(run.info.run_id)
        exit(0)

    client = mlflow.MlflowClient()
    og_run = client.get_run(args.run_id)
    if "downstream-posthoc-val-loss-mean" in og_run.data.metrics:
        print("already computed")
        exit(0)

    delta_dicts: dict = {
        "stitching": load_artifact("weights/stitching-modelAxB.pt", args.run_id),
        "downstream": load_artifact("weights/downstream-modelAxB.pt", args.run_id),
    }

    for phase, deltas in delta_dicts.items():
        delta_dict: dict = deltas
        params = og_run.data.params
        device = params["device"]

        # print(params)
        donorA = DonorSpec(params["donorA_model"], params["donorA_layer"], params["donorA_dataset"])
        donorB = DonorSpec(params["donorB_model"], params["donorB_layer"], params["donorB_dataset"])
        hybrid_model, _, _ = create_hybrid_model(
            donorA=donorA, donorB=donorB, stitch_family=params["stitch_family"]
        )
        hybrid_model.load_delta_state_dict(delta_dict, base_model=hybrid_model, strict=True)
        hybrid_model = hybrid_model.eval().to(device)

        donorB.dataset.setup("val")
        val_data = donorB.dataset.val_dataloader(batch_size=100, num_workers=4, pin_memory=True)

        loss_stats = RunningVariance()
        acc_stats = RunningVariance()

        with torch.no_grad():
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            for im, la in tqdm(val_data):
                im, la = im.to(device), la.to(device)
                out = hybrid_model(im)
                is_corrects = (torch.argmax(out, dim=-1) == la).float()
                losses = loss_fn(out, la)

                loss_stats.update(losses)
                acc_stats.update(is_corrects)

        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-loss-mean",
            value=loss_stats.mu_x.item(),
        )
        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-loss-mcse",
            value=sqrt(loss_stats.variance.item() / len(val_data.dataset)),
        )
        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-acc-mean",
            value=acc_stats.mu_x.item(),
        )
        client.log_metric(
            run_id=args.run_id,
            key=phase + "-posthoc-val-acc-mcse",
            value=sqrt(acc_stats.variance.item() / len(val_data.dataset)),
        )
