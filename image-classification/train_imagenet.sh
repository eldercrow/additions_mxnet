#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012_cls/ILSVRC2012_train.rec \
    --network dilatenetv4 \
    --image-shape '3,192,192' \
    --batch-size 192 \
    --optimizer nadam \
    --gpus 0,1 \
    --disp 200 \
    --model-prefix ./model/dilatenetv4_imagenet \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --use-plateau True \
    --lr-step-epochs 2,2,3,3,4,4 \
    --num-epoch 120
