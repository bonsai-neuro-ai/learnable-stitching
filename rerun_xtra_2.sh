#!/usr/bin/env bash
TARGETS=(
  TASK
)

MODELA="resnet18"
MODELB="resnet34"

LAYERSA=(
    "add_6"
    "add_5"
    "add_4"
)

LAYERSB=(
    "add_15"
)


    for LAYERA in "${LAYERSA[@]}"; do
      for LAYERB in "${LAYERSB[@]}"; do
        for TARGET in "${TARGETS[@]}"; do
          CUDA_VISIBLE_DEVICES=2 python3 experiment.py \
            --donorA.model="$MODELA" \
            --donorA.layer="$LAYERA" \
            --donorA.dataset="imagenet" \
            --donorB.model="$MODELB" \
            --donorB.layer="$LAYERB" \
            --donorB.dataset="imagenet" \
            --stitch_family="1x1Conv" \
            --target_type="$TARGET" \
            --init_batches=10 \
            --downstream_batches=1000 \
            --batch_size=200 \
            --num_workers=4
          done
        done
      done
  done
