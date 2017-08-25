#!/bin/bash

python demo.py \
    --network pva101 \
    --prefix ./model/ssd_pva101_384 \
    --epoch 1000 \
    --data-shape 384 \
    --gpu 0
