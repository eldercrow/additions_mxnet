#!/usr/bin/bash
# python train_imdb.py \
#     --dataset pascal_voc_patch \
#     --year 2007,2012 \
#     --network pva100_ssd_256 \
#     --data-shape 256 \
#     --val-image-set '' \
#     --from-scratch 1 \
#     --end-epoch 1

# python train_imdb.py \
#     --gpus 1 \
#     --dataset pascal_voc \
#     --year 2007,2012 \
#     --network pva100_ssd_512 \
#     --data-shape 512 \
#     --val-image-set '' \
#     --freeze '' \
#     --end-epoch 240 \
#     --resume 14
    # --pretrained ./model/ssd_256 \
    # --epoch 1
#  
python train_imdb.py \
    --gpus 1 \
    --dataset pascal_voc \
    --year 2007,2012 \
    --network pva102_ssd_512 \
    --prefix model/pva102_ssd \
    --freeze '' \
    --data-shape 512 \
    --val-image-set '' \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --lr-steps 2,2,3,3,4,4,4 \
    --frequent 100 \
    --wd 1e-04 \
    --pretrained ./model/pva102_ssd_512 \
    --epoch 1000
    # --resume 37
