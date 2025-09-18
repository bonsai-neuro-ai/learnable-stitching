from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
import mlflow.entities


def layer_idx(layer_name: str) -> int:
    """Extracts the layer index from a layer name string."""
    return int(layer_name.split("_")[-1]) if "_" in layer_name else 0


def pivot_heatmap(
    df: pd.DataFrame, row_col: str, col_col: str, metric: str, sort_key=None, ax=None, **kwargs
):
    matrix = df.pivot_table(
        index=row_col,
        columns=col_col,
        values=metric,
    )

    if sort_key is not None:
        matrix = matrix.reindex(
            sorted(matrix.index, key=sort_key),
            axis=0,
        ).reindex(
            sorted(matrix.columns, key=sort_key),
            axis=1,
        )

    sns.heatmap(matrix, annot=False, ax=ax or plt.gca(), **kwargs)


def get_metric_history(
    run_id: str,
    metric: str,
    bin_steps: int = 1,
    mlflow_client: Optional[mlflow.tracking.MlflowClient] = None,
):
    mlflow_client = mlflow_client or mlflow.tracking.MlflowClient()
    run: mlflow.entities.Run = mlflow_client.get_run(run_id)
    info = run.data.params

    def _bin_avg_helper(array):
        return [np.mean(array[s : s + bin_steps]) for s in range(0, len(array), bin_steps)]

    rows = []
    metric_history = mlflow_client.get_metric_history(run_id=run_id, key=metric)
    step_avg = _bin_avg_helper([mm.step for mm in metric_history])
    ts_avg = _bin_avg_helper([mm.timestamp for mm in metric_history])
    val_avg = _bin_avg_helper([mm.value for mm in metric_history])

    rows.extend([
        {"metric_name": metric, "step": s, "timestamp": t, "value": v, **info}
        for s, t, v in zip(step_avg, ts_avg, val_avg)
    ])

    return pd.DataFrame(rows)
