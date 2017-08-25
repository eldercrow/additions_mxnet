#!/bin/bash

python demo.py \
    --network hypernetv2 \
    --prefix ./model/ssd_hypernetv2_384 \
    --epoch 1000 \
    --images ./data/demo/dog.jpg \
    --data-shape 384 \
    --thresh 0.25 \
    --cpu
