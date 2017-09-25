#!/usr/bin/bash

python train_openimage.py \
    --data-train /home/hyunjoon/dataset/openimage_cls/train_openimage.rec \
    --network dilatenetv3 \
    --batch-size 192 \
    --image-shape '3,192,192' \
    --optimizer nadam \
    --gpus 0,1 \
    --lr 1e-03 \
    --disp 200 \
    --model-prefix ./model/dilatenetv3_openimage
    # --load-epoch 49 \
