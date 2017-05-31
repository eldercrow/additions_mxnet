#!/bin/bash
python evaluate.py \
    --dataset wider \
    --eval-set val \
    --devkit-path /home/hyunjoon/dataset/wider \
    --network spotnet_lighter \
    --prefix model/spotnet_lighter_bnfixed_768 \
    --epoch 100 \
    --gpus 0 \
    --data-shape 2560 \
    --th-pos 0.25 \
    --nms 0.333333
