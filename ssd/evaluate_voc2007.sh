#!/bin/bash
python evaluate.py \
    --dataset pascal_voc \
    --eval-set test \
    --devkit-path ./data/VOCdevkit \
    --network spotnet_384 \
    --prefix ./model/spotnet_denseconn_384 \
    --epoch 1000 \
    --gpus 0 \
    --data-shape 384 \
    --th-pos 0.25 \
    --nms 0.333333
