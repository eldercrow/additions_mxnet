#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012_cls/ILSVRC2012_train.rec \
    --network mobilenetv4 \
    --image-shape 3,224,224 \
    --batch-size 128 \
    --optimizer nadam \
    --gpus 0,1 \
    --disp 200 \
    --model-prefix ./model/mobilenetv4_imagenet \
    --lr 1e-03 \
    --lr-factor 0.1 \
    --lr-step-epochs 70,100,120 \
    --num-epoch 140
    # --use-plateau 0 \
    # --load-epoch 70 \
