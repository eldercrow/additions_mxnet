#!/bin/bash

python demo.py \
    --network dilatenetv4 \
    --prefix ./model/ssd_dilatenetv4_384 \
    --epoch 1000 \
    --images ./data/demo/dog.jpg \
    --data-shape 384 \
    --thresh 0.5 \
    --nms 0.35 \
    --cpu
