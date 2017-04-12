#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
#     --network phgnet_patch \
#     --batch-size 16 \
#     --from-scratch 1 \
#     --gpu 0 \
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
    --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
    --network phgnet \
    --batch-size 2 \
    --gpu 0 \
    --prefix model/phgnet \
    --data-shape 768 \
    --monitor 1000 \
    --frequent 50 \
    --lr 0.001 \
    --wd 0.0001 \
    --resume 1
    # --pretrained model/phgnet_patch_256 \
    # --epoch 1
