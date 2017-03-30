#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
#     --network pvtnet_preact_patch \
#     --batch-size 24 \
#     --from-scratch 1 \
#     --gpu 1 \
#     --prefix model/pvtnet_preact_patch \
#     --data-shape 256 \
#     --frequent 50 \
#     --monitor 250 \
#     --lr 0.001 \
#     --wd 0.0001

# full training
python train.py \
    --dataset wider \
    --image-set train \
    --val-image-set '' \
    --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
    --network pvtnet_preact \
    --batch-size 2 \
    --pretrained ./model/pvtnet_preact_patch_256_bnfixed \
    --epoch 0 \
    --gpu 1 \
    --prefix model/pvtnet_preact \
    --data-shape 768 \
    --monitor 200 \
    --lr 0.001 \
    --wd 0.0001
