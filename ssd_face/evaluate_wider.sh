#!/bin/bash
python evaluate.py \
    --dataset wider \
    --eval-set val \
    --devkit-path /home/hyunjoon/dataset/wider \
    --network spotnet_lite \
    --epoch 72 \
    --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/spotnet_lite2_768 \
    --gpus 0 \
    --data-shape 1280 \
    --nms 0.3 
