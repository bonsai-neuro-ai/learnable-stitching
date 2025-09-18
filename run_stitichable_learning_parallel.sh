#!/usr/bin/env bash
MODELS=(
  "resnet18"
  "resnet34"
  "resnet50"
)

TARGETS=(
  TASK
  MATCH_DOWNSTREAM
  MATCH_UPSTREAM
)

while getopts "c:t:" opt; do
  case $opt in
	c)
  	device="$OPTARG"
  	echo "Device: Cuda = $device"
  	;;
	t)
  	target_task="$OPTARG"
  	echo "TARGET: $target_task"
  	;;
  esac
done

for MODELA in "${MODELS[@]}"; do
  LAYERSA=($(python3 -m model_info $MODELA --layers | grep "add"))
  for MODELB in "${MODELS[@]}"; do
    if [ "$MODELA" = "$MODELB" ]; then continue; fi
    LAYERSB=($(python3 -m model_info $MODELB --layers | grep "add"))
    for LAYERA in "${LAYERSA[@]}"; do
      for LAYERB in "${LAYERSB[@]}"; do
          CUDA_VISIBLE_DEVICES="$device" python3 experiment.py \
            --donorA.model="$MODELA" \
            --donorA.layer="$LAYERA" \
            --donorA.dataset="imagenet" \
            --donorB.model="$MODELB" \
            --donorB.layer="$LAYERB" \
            --donorB.dataset="imagenet" \
            --stitch_family="1x1Conv" \
            --target_type="$target_task" \
            --init_batches=10 \
            --downstream_batches=1000 \
            --batch_size=200 \
            --num_workers=4
        done
      done
  done
done
