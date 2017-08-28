#!/bin/bash

python demo.py \
    --network hypernetv4 \
    --prefix ./model/ssd_hypernetv4_384 \
    --epoch 1000 \
    --images ./data/demo/street.jpg \
    --data-shape 384 \
    --thresh 0.25 \
    --nms 0.35 \
    --cpu
