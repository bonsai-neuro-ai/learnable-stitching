#!/bin/sh
# lists for the different knobs to turn for the experiments, node the deeplab models were pretrained for coco segementation"
datalist='imagenet'
modellist='resnet18 resnet34 resnet50 resnet101'

stitchlist="conv1x1" # currently hardcoded to be the 1x1 stitching layer
labellist="class" # soft labels wiil be added eventually
epochs=1 # epochs is hardcoded to 1

for data in $datalist; do
    for modelA in $modellist; do
        for modelB in $modellist; do
          if [ $modelA = $modelB ]; then continue; fi
            for stitch in $stitchlist; do
                for label in $labellist; do
                    CUDA_VISIBLE_DEVICES=1 python experiment_ver_0.1.py --dataset $data --modelA $modelA --modelB $modelB --stitch_family sf --label_type $label --epochs $epochs
                done
            done
        done
    done
done

