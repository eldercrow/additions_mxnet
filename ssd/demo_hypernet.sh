#!/bin/bash

python demo.py \
    --network hypernetv2 \
    --prefix ./model/ssd_hypernetv2_384 \
    --epoch 1000 \
    --images ./data/demo/000010.jpg \
    --data-shape 384 \
    --cpu
