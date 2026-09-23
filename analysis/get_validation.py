# %%
from os import path
import gc
import time
import dataclasses
from functools import partial
from tqdm.auto import tqdm
from typing import Self, assert_never, Literal
from nn_lib.models import get_pretrained_model
from math import isclose
import numpy as np
import csv


import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib import colormaps as cm
from nn_lib.utils import search_runs_by_params, load_artifact
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from torch.cuda import empty_cache
from torch import zeros, allclose, no_grad
from collections import defaultdict
from nn_lib.models.graph_module_plus import GraphModulePlus
from stitching import create_stitching_layer
from analysis.utils import pivot_heatmap, layer_idx, get_metric_history
from nn_lib.datasets import ImageNetDataModule, TorchvisionDataModuleBase
from graphene.utils.dataloader import DataLoader
from enum import Enum, auto
from pympler import tracker

# Take a snapshot of object growth




@dataclasses.dataclass
class DonorSpec:
    model: str | GraphModulePlus
    layer: str
    dataset: str | TorchvisionDataModuleBase

    def maybe_initialize(self) -> Self:
        if isinstance(self.model, str):
            self.model = _get_pretrained_model_by_name(self.model)
        if isinstance(self.dataset, str):
            self.dataset = _get_dataset_by_name(self.dataset)
        return self

def _get_dataset_by_name(dataset_name: str) -> TorchvisionDataModuleBase:
    # TODO - implement others and move to an nn_lib registry
    match dataset_name.lower():
        case "imagenet":
            # Imagenet has about 1.2M training images, which we'll split so that we get about 5k
            # validation images. Warning: no checks are done to ensure that the split is balanced.
            dm = ImageNetDataModule(root_dir="/data/datasets/", train_val_split=0.99)
            dm.prepare_data()
            return dm
        case _:
            assert_never(dataset_name)


def _get_pretrained_model_by_name(model_name: str) -> GraphModulePlus:
    return GraphModulePlus.new_from_trace(
        get_pretrained_model(model_name)
    ).squash_all_conv_batchnorm_pairs()

def calculate_error_bars(models, val_data, prefix, num_classes, device):
    #tr = tracker.SummaryTracker()
    #tr.print_diff()  
    """Snapshot params and run validation."""
    # Saving all three models so we can run sanity-checks later that only the parameters we
    # wanted to change actually changed in each phase.
    #for name, model in models.items():
        # Only record the keys of the state dict that have changed from the donor model. This
        # is like what we would get if we had used GraphModulePlus.delta_state_dict(), but by doing
        # it with hashes we save on space. Load these snapshots using
        # GraphModulePlus.load_delta_state_dict() and the original donor model.
        #sd_hash = hashes[name]
        #delta_state_dict = {
        #    k: v for k, v in model.state_dict().items() if _hash_tensor(v) != sd_hash[k]
        #}
        #save_as_artifact(delta_state_dict, Path("weights") / f"{prefix}-{name}.pt")

    # the metrics listed are measured on the validation set of the data
    metrics = {
        "acc1": Accuracy("multiclass", num_classes=num_classes, top_k=1).to(device),
        #"acc5": Accuracy("multiclass", num_classes=num_classes, top_k=5).to(device),
        #"loss": cross_entropy,
    }
    values = defaultdict(lambda: 0)
    means = defaultdict(lambda: 0)
    variance = defaultdict(lambda: 0)
    value_list = defaultdict(lambda: [])


    i = 1
    s = 0
    total = len(val_data)
    
    with no_grad():
        for im, la in tqdm(val_data, desc=f"Validating[{prefix}]", total=total):
            im, la = im.to(device), la.to(device)

            for model_name, model in models.items():
                out = model(im)
                for metric_name, metric_fn in metrics.items():
                    measure = metric_fn(out,la).item()
                    #if measure == 0:
                        #measure += 1e7
                    #measure = np.exp(measure)
                    
                    if i == 1:
                        m_prev = measure
                    else:
                        m_prev = means[f"{prefix}-{model_name}-val-{metric_name}"]     
                        
                    value_list[metric_name].append(measure)
                    values[f"{prefix}-{model_name}-val-{metric_name}"] += measure
                    means[f"{prefix}-{model_name}-val-{metric_name}"] += (measure) / total


                    m_current = means[f"{prefix}-{model_name}-val-{metric_name}"]
                    #s += ((measure - m_current)) * ((measure - m_prev))

                    #if k > 1:
                        #variance[f"{prefix}-{model_name}-val-{metric_name}"] = s
                        #print(variance[f"{prefix}-{model_name}-val-{metric_name}"])
            i += 1
        
                #test the running average with the original
        # Average the validation metric values over the number of batches and log to mlflow

        '''
        values = {i: v / len(val_data) for i, v in values.items()}
        test_key = ""
        for key, value in values.items():
            test_key = key
            if not (isclose(value, means[key])):
                print(f"Averages do not match for {key}: test={values[key]}\trunning={means[key]}")
                break
            else:
                print("Passed Mean Test") 

        #test the running average with the original
        # Average the validation metric values over the number of batches and log to mlflow
        values = {i: v / len(val_data) for i, v in values.items()}
        test_key = ""
        for key, value in values.items():
            test_key = key
            if not (isclose(value, means[key])):
                print(f"Averages do not match for {key}: test={values[key]}\trunning={means[key]}")
                break
            else:
                print("Passed Mean Test")    
            
    
        #test the running variance with the traditional variance
        variance_test = 0
        for x in value_list:
            variance_test += ((x - values[test_key]) * (x - values[test_key]))
        variance_test = variance_test / (total - 1)
        
        
        

        #if not (isclose(np.var(value_list), variance[test_key] / (total-1))):
            #print(f"variances do not match for {test_key}: test={variance_test}\trunning={variance[test_key] / (total-1)}")
        #else:
            #print("Passed Variance Test")'''      
    return means, variance, value_list
    



    #mlflow.log_metrics(values)
def create_hybrid_model(
    donorA: DonorSpec, donorB: DonorSpec, stitch_family: str
) -> tuple[GraphModulePlus, GraphModulePlus, GraphModulePlus]:
    """Initializes a stitched model, inferring the shapes of the layers to be stitched.

    :param donorA: A DonorSpec object representing the upstream model & layer to be stitched.
    :param donorB: A DonorSpec object representing the downstream model & layer to be stitched.
    :param stitch_family: A string passed to create_stitching_layer specifying the type of
        stitching layer.
    :return: A tuple of
        - modelAxB: The stitched model.
        - donorA_embedding_getter: A GraphModulePlus object for extracting embeddings from donorA.
        - donorB_embedding_getter: A GraphModulePlus object for extracting embeddings from donorB.
    """

    # Set up models and data if not already initialized
    donorA.maybe_initialize()
    donorB.maybe_initialize()

    image = donorA.model.to_dot().create_png(prog="dot")
    dir_path = path.dirname(path.realpath(__file__))
    with open("donorA-computation-map.png", "wb") as f:
        f.write(image)

    # Run dummy data through modelA up to layerA to get its shape
    donorA_embedding_getter = GraphModulePlus.new_from_copy(donorA.model).set_output(donorA.layer)
    dataA_shape = donorA.dataset.shape
    repA_shape = donorA_embedding_getter(zeros((1,) + dataA_shape)).shape

    # Run dummy data through modelB up to layerB to get its shape
    donorB_embedding_getter = GraphModulePlus.new_from_copy(donorB.model).set_output(donorB.layer)
    dataB_shape = donorB.dataset.shape
    repB_shape = donorB_embedding_getter(zeros((1,) + dataB_shape)).shape

    # Initialize the stitching layer
    stitching_layer = create_stitching_layer(repA_shape[1:], repB_shape[1:], stitch_family)

    # sanity-check that the stiching layer is the correct size
    assert stitching_layer(zeros(repA_shape)).shape == repB_shape

    # retrieve next node for stitching the graph
    layerB_next = donorB.model.users_of(donorB.layer)
    assert len(layerB_next) == 1, "Stitching layers with |users| != 1 not supported"
    layerB_next = layerB_next[0].name

    # Stitch the stitching layer to the bottom of A_ subgraph
    modelAxB = GraphModulePlus.new_from_merge(
        {"donorA": donorA.model, "stitching_layer": stitching_layer, "donorB": donorB.model},
        {
            "stitching_layer": ["donorA_" + donorA.layer],
            "donorB_" + layerB_next: ["stitching_layer"],
        },
        auto_trace=False,
    )

    
    image = modelAxB.to_dot().create_png(prog="dot")
    with open("hybrid-computation-map.png", "wb") as f:
        f.write(image)
    

    return modelAxB, donorA_embedding_getter, donorB_embedding_getter


##=================================================================================##
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
            run_id = str(sys.argv[1])
    else:
        print("No run_id provided")
        exit(0)

    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
    mlflow.set_experiment("learnable-stitching-v0.4")
    client = mlflow.MlflowClient()



    delta_dicts: dict = {"stitching" : load_artifact("weights/stitching-modelAxB.pt", run_id), "downstream": load_artifact("weights/downstream-modelAxB.pt", run_id)}
    
    for phase, deltas in delta_dicts.items():
        delta_dict: dict = deltas
        params = client.get_run(run_id).data.params

        #print(params)
        donorA = DonorSpec(
                params["donorA_model"],
                params["donorA_layer"],
                params["donorA_dataset"]
            )
        
        donorB = DonorSpec(
                params["donorB_model"],
                params["donorB_layer"],
                params["donorB_dataset"]
            )
        hybrid_model, _ , _ = create_hybrid_model (
            donorA=donorA,
            donorB = donorB,
            stitch_family=params["stitch_family"]
        )

        hybrid_model.load_delta_state_dict(delta_dict, base_model=hybrid_model, strict=True)

        hybrid_model = hybrid_model.eval().to(params["device"])

        # set up run name
        
        match str(params["target_type"]):
            case "TargetType.TASK":
                task_name = "task"
            case "TargetType.MATCH_UPSTREAM":
                task_name = "upstream"
            case "TargetType.MATCH_DOWNSTREAM":
                task_name = "downstream"
            case _:
                assert_never(params["target_type"])

        run_name = (
            params["donorA_model"]
            + "_"
            + params["donorA_layer"]
            + "_"
            + params["donorA_dataset"]
            + "-X-"
            + params["donorB_model"]
            + "_"
            + params["donorB_layer"]
            + "_"
            + params["donorB_dataset"]
            + "-"
            + params["stitch_family"]
            + "_"
            + task_name
        )

        donorB.dataset.setup("val")
        val_data = donorB.dataset.val_dataloader(
            batch_size=1, num_workers=4, drop_last=True
        )

        print(len(val_data))
        num_data = len(val_data)

        means, variance, value_list = calculate_error_bars(models={run_name : hybrid_model}, val_data=val_data, prefix="test", num_classes=donorB.dataset.num_classes, device=params["device"])
        
        for k, v in means.items():
            mean = v
            model_name = k.split('-')[1]
            metric_name = k.split('-')[-1]

            value_np = np.array(value_list[metric_name])
            variance[k] = np.var(value_np)

            new_row = [run_id, model_name, metric_name, phase, mean, variance[k], num_data]

            with open('analysis/acc.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_row)
    