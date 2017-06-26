#!/bin/bash
python evaluate.py \
    --dataset wider \
    --eval-set val \
    --devkit-path /home/hyunjoon/dataset/wider \
    --network spotnet_lighter3 \
    --prefix model/spotnet_lighter3_bnfixed_768 \
    --epoch 1000 \
    --gpus 1 \
    --data-shape 2560 \
    --th-pos 0.25 \
    --nms 0.333333
