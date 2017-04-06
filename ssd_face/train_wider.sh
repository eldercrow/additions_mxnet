#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
#     --network hjnet_preact_patch \
#     --batch-size 16 \
#     --from-scratch 1 \
#     --gpu 0 \
#     --prefix model/hjnet_preact_patch \
#     --data-shape 256 \
#     --frequent 20 \
#     --monitor 500 \
#     --lr 0.001 \
#     --wd 0.0001

# full training
python train.py \
    --dataset wider \
    --image-set train \
    --val-image-set '' \
    --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
    --network hjnet_preact \
    --batch-size 2 \
    --gpu 0 \
    --prefix model/hjnet_preact \
    --data-shape 768 \
    --monitor 1000 \
    --frequent 50 \
    --lr 0.0001 \
    --wd 0.0001 \
    --resume 23
    # --pretrained model/hjnet_preact_patch_256 \
    # --epoch 0 
