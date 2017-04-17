#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/dataset/wider \
#     --network phgnet_patch \
#     --batch-size 16 \
#     --from-scratch 1 \
#     --gpu 1 \
#     --prefix model/phgnet_patch \
#     --data-shape 256 \
#     --end-epoch 1 \
#     --frequent 50 \
#     --monitor 1000 \
#     --lr 0.001 \
#     --wd 0.0001

# full training
python train.py \
    --dataset wider \
    --image-set train \
    --val-image-set '' \
    --devkit-path /home/hyunjoon/dataset/wider \
    --network phgnet \
    --batch-size 2 \
    --gpu 0 \
    --prefix model/phgnet \
    --data-shape 768 \
    --monitor 1000 \
    --frequent 50 \
    --lr 0.001 \
    --lr-steps 25,40,50 \
    --lr-factor 0.316228 \
    --wd 0.0001 \
    --pretrained model/phgnet_patch_256 \
    --epoch 1
    # --resume 1
