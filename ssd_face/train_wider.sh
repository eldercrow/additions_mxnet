#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
#     --network hjnet_preact_patch \
#     --batch-size 24 \
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
    --batch-size 3 \
    --pretrained model/hjnet_preact_patch_256 \
    --epoch 0 \
    --gpu 0 \
    --prefix model/hjnet_preact \
    --data-shape 768 \
    --monitor 200 \
    --lr 0.001 \
    --wd 0.0001
