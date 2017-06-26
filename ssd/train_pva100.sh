#!/usr/bin/bash
python train_imdb.py \
    --dataset pascal_voc_patch \
    --network pva100_ssd_512 \
    --data-shape 256 \
    --val-image-set ''
