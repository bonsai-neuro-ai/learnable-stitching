#!/usr/bin/env bash
TARGETS=(
  TASK
)

MODELB="fcn_resnet50"
MODELA="resnet18"


LAYERSA=(
    "add_1"
)

LAYERSB=(
    "add_9"
)

for LAYERA in "${LAYERSA[@]}"; do
  for LAYERB in "${LAYERSB[@]}"; do
    for TARGET in "${TARGETS[@]}"; do
      CUDA_VISIBLE_DEVICES=1 python3 experiment.py \
        --donorA.model="$MODELA" \
        --donorA.layer="$LAYERA" \
        --donorA.dataset="imagenet" \
        --donorB.model="$MODELB" \
        --donorB.layer="$LAYERB" \
        --donorB.dataset="coco" \
        --stitch_family="1x1Conv" \
        --finetune_init="regression+stitch" \
        --init_batches=100 \
        --batch_size=75 \
        --num_workers=4 || exit 1
      done
    done
  done