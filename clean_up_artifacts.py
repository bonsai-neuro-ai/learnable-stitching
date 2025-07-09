from copy import deepcopy

import mlflow
import argparse
from pathlib import Path
import torch
import pandas as pd
from nn_lib.models import GraphModulePlus

# Import stuff from experiment_ver_0.2.py (TODO: remove dot from the filename)
import importlib.util

from torch import nn

spec = importlib.util.spec_from_file_location("experiment_ver_0_2", "experiment_ver_0.2.py")
experiment_src = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_src)

TargetType = experiment_src.TargetType
DonorSpec = experiment_src.DonorSpec
_get_pretrained_model_by_name = experiment_src._get_pretrained_model_by_name
create_hybrid_model = experiment_src.create_hybrid_model


def regenerate_models_fron_run(run: pd.Series):
    donorA = DonorSpec(
        model=run["params.donorA_model"],
        layer=run["params.donorA_layer"],
        dataset=run["params.donorA_dataset"],
    )
    donorB = DonorSpec(
        model=run["params.donorB_model"],
        layer=run["params.donorB_layer"],
        dataset=run["params.donorB_dataset"],
    )

    donorA.maybe_initialize()
    donorB.maybe_initialize()
    modelAxB, _, _ = create_hybrid_model(donorA, donorB, stitch_family=run["params.stitch_family"])

    modelA = donorA.model.eval()
    modelB = donorB.model.eval()
    modelAxB = modelAxB.eval()

    return modelA, modelB, modelAxB


def load_copy(model: nn.Module, checkpoint_file: Path, map_location="cpu") -> nn.Module:
    copy_model = deepcopy(model)
    copy_model.load_state_dict(torch.load(checkpoint_file, map_location=map_location))
    return copy_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up MLflow artifacts")
    parser.add_argument("--experiment", type=str, required=True, help="MLflow experiment name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not modify any files, just print what would be done",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns/")
    runs = mlflow.search_runs(experiment_names=[args.experiment])
    print("Cleaning up", len(runs), "runs")
    for idx, run in runs.iterrows():
        baseA, baseB, baseAxB = regenerate_models_fron_run(run)

        artifact_dir = Path(run["artifact_uri"]).resolve()
        for checkpoint_file in artifact_dir.glob("weights/*.pt"):
            if "modelAxB" in checkpoint_file.name:
                og_model = baseAxB
            elif "modelA" in checkpoint_file.name:
                og_model = baseA
            elif "modelB" in checkpoint_file.name:
                og_model = baseB
            else:
                print(f"Skipping unknown checkpoint file: {checkpoint_file}")
                continue

            new_model = load_copy(og_model, checkpoint_file, map_location="cpu")
            delta = new_model.delta_state_dict(og_model)
            if args.dry_run:
                print(
                    checkpoint_file,
                    "would replace file with",
                    len(og_model.state_dict()),
                    "keys with",
                    len(delta),
                    "keys",
                )
            else:
                # WARNING: This will overwrite the original checkpoint file
                torch.save(delta, checkpoint_file)
