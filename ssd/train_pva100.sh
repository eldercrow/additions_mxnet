#!/usr/bin/bash
python train_imdb.py \
    --dataset pascal_voc_patch \
    --year 2007,2012 \
    --network pva100_ssd_256 \
    --data-shape 256 \
    --val-image-set '' \
    --from-scratch 1 \
    --end-epoch 1
