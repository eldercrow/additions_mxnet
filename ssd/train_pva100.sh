#!/usr/bin/bash
python train_imdb.py \
    --network pva100_ssd_512 \
    --data-shape 512 \
    --val-image-set ''
