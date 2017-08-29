#!/bin/bash

python demo.py \
    --network pva101 \
    --prefix ./model/ssd_pva101_384 \
    --epoch 40 \
    --images ./data/demo/street.jpg \
    --data-shape 384 \
    --thresh 0.5 \
    --cpu
