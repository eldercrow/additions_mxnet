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
    --network pva102_ssd_512 \
    --prefix model/pva102_ssd \
    --freeze '' \
    --batch-size 16 \
    --data-shape 512 \
    --val-image-set '' \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --lr-steps 2,2,3,3,4,4,4 \
    --frequent 50 \
    --wd 1e-04 \
    --pretrained ./model/pva102_ssd_512 \
    --epoch 1000
    # --resume 37
