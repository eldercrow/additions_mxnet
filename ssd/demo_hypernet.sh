#!/bin/bash

python demo.py \
    --network hypernetv3 \
    --prefix ./model/ssd_hypernetv3_384 \
    --epoch 1000 \
    --images ./data/demo/street.jpg \
    --data-shape 384 \
    --thresh 0.5 \
    --cpu
