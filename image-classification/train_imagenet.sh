#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012_cls/ILSVRC2012_train.rec \
    --network mobilenetv6 \
    --image-shape 3,224,224 \
    --min-random-scale 0.875 \
    --max-random-scale 1.143 \
    --batch-size 256 \
    --optimizer sgd \
    --gpus 0,1 \
    --disp 100 \
    --model-prefix ./model/mobilenetv6_imagenet \
    --lr 0.05 \
    --wd 5e-05 \
    --lr-factor 0.1 \
    --lr-step-epochs 75,120 \
    --num-epoch 150
    # --use-plateau 1 \
    # --load-epoch 49 \
