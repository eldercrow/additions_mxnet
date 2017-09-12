#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012/ILSVRC2012_train.rec \
    --network hypernetv3 \
    --optimizer nadam \
    --gpus 0,1 \
    --disp 200 \
    --model-prefix ./model/hypernetv3 \
    --use-plateau '' \
    --load-epoch 26 \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --lr-step-epochs 40,60,80,100 \
    --num-epoch 120
