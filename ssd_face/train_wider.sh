#!/bin/bash
# from scratch
# python train.py \
#     --dataset wider_patch \
#     --image-set train \
#     --val-image-set '' \
#     --devkit-path /home/hyunjoon/dataset/wider \
#     --network spotnet_sep_patch \
#     --batch-size 16 \
#     --from-scratch 1 \
#     --gpu 1 \
#     --prefix model/spotnet_sep_patch \
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
    --network spotnet_lite_bnfixed \
    --batch-size 4 \
    --gpu 1 \
    --prefix model/spotnet_lite2_bnfixed_768 \
    --data-shape 768 \
    --frequent 100 \
    --lr 1e-05 \
    --lr-steps 10,20,30,40,45,50 \
    --lr-factor 0.316228 \
    --wd 1e-05 \
    --pretrained model/spotnet_lite2_bnfixed_768 \
    --epoch 0
    # --resume 0
    # --monitor 2000 \
