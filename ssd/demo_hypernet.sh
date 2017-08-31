#!/bin/bash

python demo.py \
    --network hypernetv5 \
    --prefix ./model/ssd_hypernetv5_384 \
    --epoch 1000 \
    --images ./data/demo/000010.jpg \
    --data-shape 384 \
    --thresh 0.5 \
    --nms 0.45 \
    --cpu
