#!/usr/bin/bash
# python train_imdb.py \
#     --dataset pascal_voc_patch \
#     --year 2007,2012 \
#     --network spotnet_256 \
#     --prefix model/spotnet \
#     --lr 0.001 \
#     --data-shape 256 \
#     --val-image-set '' \
#     --from-scratch 1 \
#     --end-epoch 5

python train_imdb.py \
    --gpus 1 \
    --dataset pascal_voc \
    --year 2007,2012 \
    --network spotnet_512 \
    --prefix model/spotnet \
    --freeze '' \
    --batch-size 16 \
    --data-shape 512 \
    --val-image-set '' \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --lr-steps 2,2,3,3,4,4,4 \
    --frequent 50 \
    --wd 1e-04 \
    --resume 18
    # --pretrained ./model/spotnet_256 \
    # --epoch 1000
