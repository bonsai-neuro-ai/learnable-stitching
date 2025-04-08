#!/bin/sh
modellist='resnet18 resnet34 resnet50'

for modelA in $modellist; do
  for modelB in $modellist; do
    if [ $modelA = $modelB ]; then continue; fi
    CUDA_VISIBLE_DEVICES=1 python experiment_ver_0.1.py \
      --modelA $modelA \
      --modelB $modelB \
      --batch_size=200 \
      --stitch_family="1x1Conv" \
      --label_type="class" \
      --init_batches=10 \
      --downstream_batches=100 \
      --batch_size=200 \
      --num_workers=20
  done
done
