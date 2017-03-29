#!/bin/bash
# python train.py \
#     --dataset wider \
#     --image-set train \
#     --val-image-set val \
#     --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
#     --network hjnet \
#     --batch-size 16 \
#     --pretrained /home/hyunjoon/github/model_mxnet/pva910/pvanet_bn_freezed \
#     --epoch 0 \
#     --gpu 0 \
#     --prefix hjnet_face \
#     --data-shape 200 \
#     --monitor 1000 \
#     --lr 0.001 \
#     --wd 0.0001
    # --cpu 1

# from scratch
python train.py \
    --dataset wider \
    --image-set train \
    --val-image-set '' \
    --devkit-path /home/hyunjoon/fd/joint_cascade/data/wider \
    --network pvtnet_preact_patch \
    --batch-size 24 \
    --from-scratch 1 \
    --gpu 0 \
    --prefix model/pvtnet_preact_patch \
    --data-shape 256 \
    --frequent 50 \
    --monitor 250 \
    --lr 0.001 \
    --wd 0.0001
