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
#     --gpu 0 \
#     --prefix model/spotnet_lighter3_patch \
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
    --network spotnet_lighter3 \
    --batch-size 4 \
    --gpu 0 \
    --prefix model/spotnet_lighter3_bnfixed \
    --data-shape 768 \
    --frequent 800 \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --lr-steps 2,2,3,3,4,4 \
    --wd 1e-04 \
    --pretrained model/spotnet_lighter3_bnfixed_768 \
    --epoch 1000
    # --resume 1
    # --monitor 2000 \
    # --lr-steps 10,15,18,21 \
