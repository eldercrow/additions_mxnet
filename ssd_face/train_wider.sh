#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/dataset/wider \
#     --network spotnet_x2v2_patch \
#     --batch-size 16 \
#     --from-scratch 1 \
#     --gpu 1 \
#     --prefix model/spotnet_x2v2_patch \
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
    --network spotnet_lighter_bnfixed \
    --batch-size 2 \
    --gpu 1 \
    --prefix model/spotnet_lighter_bnfixed \
    --data-shape 768 \
    --frequent 200 \
    --lr 1e-04 \
    --lr-factor 0.316228 \
    --lr-steps 2,2,3,3 \
    --wd 1e-05 \
    --pretrained model/spotnet_lighter_bnfixed_768 \
    --epoch 0
    # --lr-steps 10,15,18,21 \
    # --resume 1
    # --monitor 2000 \
