#!/bin/bash

python evaluate.py \
    --network hypernetv2 \
    --data-shape 384 \
    --prefix ./model/ssd_hypernetv2 \
    --epoch 1000 \
    --batch-size 16 \
    --nms-thresh 0.45 \
    --gpus 0
