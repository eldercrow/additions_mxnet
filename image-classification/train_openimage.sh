#!/usr/bin/bash

python train_openimage.py \
    --data-train /home/hyunjoon/dataset/openimage/rec_classification/train_openimage.rec \
    --network hypernetv6 \
    --batch-size 192 \
    --image-shape '3,192,192' \
    --optimizer nadam \
    --gpus 0,1 \
    --lr 1e-03 \
    --disp 200 \
    --load-epoch 49 \
    --model-prefix ./model/hypernetv6_openimage
