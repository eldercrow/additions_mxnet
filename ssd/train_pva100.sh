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
#     --pretrained ./model/ssd_256 \
#     --epoch 1
#     # --from-scratch 1
 
python train_imdb.py \
    --gpus 0 \
    --dataset pascal_voc \
    --year 2007,2012 \
    --network pva101_ssd_512 \
    --data-shape 512 \
    --val-image-set '' \
    --end-epoch 240 \
    --pretrained /home/hyunjoon/github/model_mxnet/pva101/pva101 \
    --epoch 0
