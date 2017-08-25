#!/bin/bash

python demo.py \
    --network pva101 \
    --prefix ./model/ssd_pva101_384 \
    --epoch 1000 \
    --images ./data/demo/dog.jpg \
    --data-shape 384 \
    --thresh 0.25 \
    --cpu
