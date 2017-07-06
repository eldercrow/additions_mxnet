#!/bin/bash
python evaluate.py \
    --dataset pascal_voc \
    --eval-set test \
    --devkit-path ./data/VOCdevkit \
    --network spotnet_512 \
    --prefix ./model/spotnet_512 \
    --epoch 1000 \
    --gpus 0 \
    --data-shape 960 \
    --th-pos 0.25 \
    --nms 0.333333
