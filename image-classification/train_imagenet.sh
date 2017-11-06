#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012/ILSVRC2012_train.rec \
    --network mobilenetv3 \
    --image-shape '3,192,192' \
    --batch-size 192 \
    --optimizer nadam \
    --gpus 0,1 \
    --disp 200 \
    --model-prefix ./model/mobilenetv3_imagenet \
    --lr 1e-03 \
    --lr-factor 0.1 \
    --load-epoch 70 \
    --lr-step-epochs 70,100,120 \
    --num-epoch 140
    # --use-plateau 0 \
