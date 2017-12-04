#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012_cls/ILSVRC2012_train.rec \
    --network mobilenetv8 \
    --image-shape '3,224,224' \
    --max-random-scale 1.143 \
    --min-random-scale 0.875 \
    --batch-size 256 \
    --optimizer sgdnadam \
    --gpus 0,1 \
    --disp 100 \
    --model-prefix ./model/mobilenetv8_imagenet \
    --lr 0.05 \
    --wd 5e-05 \
    --lr-factor 0.1 \
    --lr-step-epochs 75,120 \
    --num-epoch 150
    # --use-plateau 1 \
    # --load-epoch 49 \
