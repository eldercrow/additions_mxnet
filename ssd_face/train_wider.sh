#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/dataset/wider \
#     --network spotnet_lighter_patch \
#     --batch-size 16 \
#     --from-scratch 1 \
#     --gpu 1 \
#     --prefix model/spotnet_lighter2_patch \
#     --data-shape 256 \
#     --end-epoch 1 \
#     --frequent 50 \
#     --lr 1e-03 \
#     --wd 1e-04
    # --monitor 1000 \

# full training
python train.py \
    --dataset wider \
    --image-set train \
    --val-image-set '' \
    --devkit-path /home/hyunjoon/dataset/wider \
    --network spotnet_lighter_bnfixed \
    --batch-size 4 \
    --gpu 1 \
    --prefix model/spotnet_lighter2_clonefixed \
    --data-shape 768 \
    --frequent 200 \
    --lr 1e-04 \
    --lr-factor 0.1 \
    --lr-steps 3,3 \
    --wd 1e-04 \
    --resume 11 
    # --pretrained model/spotnet_lighter2_bnfixed_768 \
    # --epoch 0
    # --monitor 2000 \
    # --lr-steps 10,15,18,21 \
