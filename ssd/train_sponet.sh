#!/usr/bin/bash
# python train_imdb.py \
#     --gpus 0 \
#     --dataset pascal_voc_patch \
#     --year 2007,2012 \
#     --network spotnet_256 \
#     --frequent 100 \
#     --prefix model/spotnet \
#     --lr 0.001 \
#     --data-shape 256 \
#     --val-image-set '' \
#     --from-scratch 1 \
#     --end-epoch 5

python train_imdb.py \
    --gpus 0 \
    --dataset pascal_voc \
    --year 2007,2012 \
    --network spotnet_512 \
    --prefix model/spotnet \
    --freeze '' \
    --batch-size 16 \
    --data-shape 384 \
    --val-image-set '' \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --lr-steps 2,2,3,3,4,4,4 \
    --frequent 100 \
    --wd 1e-04 \
    --pretrained ./model/spotnet_256 \
    --epoch 1000
#     --resume 31
