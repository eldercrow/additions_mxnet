#!/usr/bin/bash

python train_openimage.py \
    --data-train /home/hyunjoon/dataset/openimage_cls/train_openimage.rec \
    --network dilatenetv4 \
    --batch-size 192 \
    --image-shape '3,192,192' \
    --optimizer nadam \
    --gpus 0,1 \
    --lr 1e-03 \
    --use-plateau True \
    --lr-step-epochs 2,2,3,3,4,4 \
    --lr-factor 0.316227766 \
    --disp 200 \
    --model-prefix ./model/dilatenetv4_openimage
    # --load-epoch 49 \
