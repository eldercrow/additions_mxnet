#!/bin/bash
python evaluate.py \
    --dataset pascal_voc \
    --eval-set test \
    --devkit-path ./data/VOCdevkit \
    --network ron_480 \
    --prefix ./model/ron_voc_480 \
    --epoch 1000 \
    --gpus 0 \
    --data-shape 960 \
    --th-pos 0.25 \
    --nms 0.333333
