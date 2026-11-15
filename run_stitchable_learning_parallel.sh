#!/usr/bin/env bash
MODELSA=(
  "fcn_resnet50"
)

MODELSB=(
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
  LAYERSA=($(python3 -m model_info $MODELA --layers "odd" | grep "add"))
  for MODELB in "${MODELSB[@]}"; do
    if [ "$MODELA" = "$MODELB" ]; then continue; fi
    LAYERSB=($(python3 -m model_info $MODELB --layers "odd" | grep "add"))
    for LAYERA in "${LAYERSA[@]}"; do
      for LAYERB in "${LAYERSB[@]}"; do
          CUDA_VISIBLE_DEVICES="$device" python3 experiment.py \
            --donorA.model="$MODELA" \
            --donorA.layer="$LAYERA" \
            --donorA.dataset="coco" \
            --donorB.model="$MODELB" \
            --donorB.layer="$LAYERB" \
            --donorB.dataset="imagenet" \
            --stitch_family="1x1Conv" \
            --finetune_init="$stitch_init" \
            --batch_size=200 \
            --num_workers=4 \
            --mlflow_experiment="learnable-stitching-v0.5" || exit 1
        done
      done
  done
done
