#!/usr/bin/bash
# python train_imdb.py \
#     --dataset pascal_voc_patch \
#     --year 2007,2012 \
#     --network spotnet_256 \
#     --prefix model/spotnet \
#     --data-shape 256 \
#     --val-image-set '' \
#     --from-scratch 1 \
#     --end-epoch 2

python train_imdb.py \
    --gpus 1 \
    --dataset pascal_voc \
    --year 2007,2012 \
    --network spotnet_512 \
    --prefix model/spotnet_ssd \
    --freeze '' \
    --batch-size 8 \
    --data-shape 512 \
    --val-image-set '' \
    --lr 0.000316227766 \
    --lr-factor 0.316227766 \
    --lr-steps 2,2,3,3,4,4,4 \
    --frequent 50 \
    --wd 1e-04 \
    --resume 8
    # --pretrained ./model/spotnet_ssd_512 \
    # --epoch 1000
