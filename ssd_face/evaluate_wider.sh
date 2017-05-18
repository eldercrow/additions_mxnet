#!/bin/bash
python evaluate.py \
    --dataset wider \
    --eval-set val \
    --devkit-path /home/hyunjoon/dataset/wider \
    --network spotnet_lighter_bnfixed \
    --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/spotnet_lighter_bnfixed_768 \
    --epoch 50 \
    --gpus 0 \
    --data-shape 1440 \
    --th-pos 0.25 \
    --nms 0.33333 
