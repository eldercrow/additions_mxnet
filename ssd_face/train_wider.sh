#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/dataset/wider \
#     --network spotnet_patch \
#     --batch-size 16 \
#     --from-scratch 1 \
#     --gpu 1 \
#     --prefix model/spotnet_patch \
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
    --network spotnet_xy \
    --batch-size 2 \
    --gpu 1 \
    --prefix model/spotnet_xy \
    --data-shape 768 \
    --frequent 50 \
    --lr 0.001 \
    --lr-steps 10,20,25,30,35,40 \
    --lr-factor 0.316228 \
    --wd 1e-05 \
    --pretrained model/spotnet_xy_768 \
    --epoch 0
    # --resume 0
    # --monitor 200 \
