#!/bin/bash
python evaluate.py \
    --dataset pascal_voc \
    --eval-set test \
    --devkit-path ./data/VOCdevkit \
    --network pva102_ssd_512 \
    --prefix ./model/pva102_ssd_512 \
    --epoch 1000 \
    --gpus 0 \
    --data-shape 960 \
    --th-pos 0.25 \
    --nms 0.333333
