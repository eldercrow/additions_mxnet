#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012_cls/ILSVRC2012_train.rec \
    --network mobilenetv6 \
    --image-shape '3,224,224' \
    --max-random-scale 1.143 \
    --min-random-scale 0.875 \
    --batch-size 256 \
    --optimizer sgd \
    --gpus 0,1 \
    --disp 100 \
    --model-prefix ./model/mobilenetv6_imagenet \
    --lr 5e-02 \
    --lr-factor 0.1 \
    --wd 2e-05 \
    --lr-step-epochs 70,100,120 \
    --load-epoch 112 \
    --num-epoch 140
    # --load-epoch 14 \
    # --use-plateau 0 \
