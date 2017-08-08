#!/usr/bin/bash
# python train_imdb.py \
#     --gpus 1 \
#     --dataset pascal_voc_patch \
#     --year 2007,2012 \
#     --network spotnet_256 \
#     --frequent 100 \
#     --prefix model/spotnet_voc \
#     --lr 0.001 \
#     --batch-size 24 \
#     --data-shape 256 \
#     --frequent 100 \
#     --val-image-set '' \
#     --from-scratch 1 \
#     --end-epoch 5
#
python train_imdb.py \
    --gpus 1 \
    --dataset pascal_voc \
    --year 2007,2012 \
    --network spotnet_480 \
    --prefix model/spotnet_multibox \
    --freeze '' \
    --batch-size 12 \
    --data-shape 480 \
    --val-image-set '' \
    --lr 1e-04 \
    --lr-factor 0.316227766 \
    --lr-steps 3,3,4,4,4 \
    --frequent 100 \
    --wd 5e-04 \
    --pretrained ./model/spotnet_multibox_480 \
    --epoch 1000
    # --resume 59
