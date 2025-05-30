#!/usr/bin/env bash
MODELS=(
  "resnet18"
  "resnet34"
  "resnet50"
)

for MODELA in "${MODELS[@]}"; do
  LAYERSA=($(python -m model_info $MODELA --layers | grep "add"))
  for MODELB in "${MODELS[@]}"; do
    if [ "$MODELA" = "$MODELB" ]; then continue; fi
    LAYERSB=($(python -m model_info $MODELB --layers | grep "add"))
    for LAYERA in "${LAYERSA[@]}"; do
      for LAYERB in "${LAYERSA[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python experiment_ver_0.2.py \
          --donorA.model="$MODELA" \
          --donorA.layer="$LAYERA" \
          --donorA.dataset="imagenet" \
          --donorB.model="$MODELB" \
          --donorB.layer="$LAYERB" \
          --donorB.dataset="imagenet" \
          --stitch_family="1x1Conv" \
          --target_type=TASK \
          --init_batches=10 \
          --downstream_batches=500 \
          --batch_size=200 \
          --num_workers=4
        done
      done
  done
done
