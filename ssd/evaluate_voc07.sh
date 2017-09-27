#!/bin/bash

python evaluate.py \
    --network dilatenetv4 \
    --rec-path ./data/VOCdevkit/val.rec \
    --data-shape 384 \
    --prefix ./model/ssd_dilatenetv4 \
    --epoch 1000 \
    --batch-size 16 \
    --nms-thresh 0.45 \
    --gpus 0
