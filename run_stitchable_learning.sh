#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "resnet34"
  "resnet50"
)

TARGETS=(
  TASK
  MATCH_DOWNSTREAM
  MATCH_UPSTREAM
)

for MODELA in "${MODELS[@]}"; do
  LAYERSA=($(python3 -m model_info $MODELA --layers | grep "add"))
  for MODELB in "${MODELS[@]}"; do
    if [ "$MODELA" = "$MODELB" ]; then continue; fi
    LAYERSB=($(python3 -m model_info $MODELB --layers | grep "add"))
    for LAYERA in "${LAYERSA[@]}"; do
      for LAYERB in "${LAYERSB[@]}"; do
        for TARGET in "${TARGETS[@]}"; do
          CUDA_VISIBLE_DEVICES=0 python3 experiment.py \
            --donorA.model="$MODELA" \
            --donorA.layer="$LAYERA" \
            --donorA.dataset="imagenet" \
            --donorB.model="$MODELB" \
            --donorB.layer="$LAYERB" \
            --donorB.dataset="imagenet" \
            --config=configs/experiment_better_convergence.yaml
          done
        done
      done
  done
done
