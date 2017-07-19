#!/usr/bin/bash

python train_openimage.py \
    --data-train /media/hyunjoon/dataset/openimage/rec/train_openimage.rec \
    --network pva101 \
    --optimizer adam \
    --gpus 0 \
    --lr 1e-03 \
    --disp 200 \
    --load-epoch 1 \
    --model-prefix ./model/pva101_pretrained
