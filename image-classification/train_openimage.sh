#!/usr/bin/bash

python train_openimage.py \
    --data-train /home/hyunjoon/dataset/openimage/rec_classification/train_openimage.rec \
    --network hypernet \
    --batch-size 192 \
    --optimizer nadam \
    --gpus 0,1 \
    --lr 1e-03 \
    --disp 200 \
    --model-prefix ./model/hypernet_openimage
    # --load-epoch 1 \
