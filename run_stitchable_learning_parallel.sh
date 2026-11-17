#!/usr/bin/env bash
MODELSB=(
  "fcn_resnet50"
)

MODELSA=(
  "resnet18"
  "resnet34"
  "resnet50"
)

TARGETS=(
  TASK
  MATCH_DOWNSTREAM
  MATCH_UPSTREAM
)

while getopts "c:t:s:" opt; do
  case $opt in
	c)
  	device="$OPTARG"
  	echo "Device: Cuda = $device"
  	;;
	t)
  	target_task="$OPTARG"
  	echo "TARGET: $target_task"
  	;;
	s)
  	stitch_init="$OPTARG"
  	echo "stitch init: $stitch_init"
  	;;
  esac
done

for MODELA in "${MODELSA[@]}"; do
  LAYERSA=($(python3 -m model_info $MODELA | grep "add" | awk 'NR % 2 == 0'))
  for MODELB in "${MODELSB[@]}"; do
    if [ "$MODELA" = "$MODELB" ]; then continue; fi
    LAYERSB=($(python3 -m model_info $MODELB | grep "add" | awk 'NR % 2 == 0'))
    for LAYERA in "${LAYERSA[@]}"; do
      for LAYERB in "${LAYERSB[@]}"; do
          CUDA_VISIBLE_DEVICES="$device" python3 experiment.py \
            --donorA.model="$MODELA" \
            --donorA.layer="$LAYERA" \
            --donorA.dataset="imagenet" \
            --donorB.model="$MODELB" \
            --donorB.layer="$LAYERB" \
            --donorB.dataset="coco" \
            --stitch_family="1x1Conv" \
            --finetune_init="$stitch_init" \
            --batch_size=75 \
            --num_workers=4 \
            --mlflow_experiment="learnable-stitching-v0.5"
        done
      done
  done
done
