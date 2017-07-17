#!/usr/bin/bash

python train_imagenet.py \
    --data-train ~/dataset/ILSVRC2012/ILSVRC2012_train.rec \
    --network spotnet_face_clone \
    --optimizer adam \
    --gpus 0 \
    --lr 1e-03 \
    --disp 200 \
    --model-prefix ./model/spotnet_pretrained
