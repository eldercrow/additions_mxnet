#!/bin/bash

python demo.py \
    --network dilatenetv2 \
    --prefix ./model/ssd_dilatenetv2_384 \
    --epoch 1000 \
    --images ./data/demo/street.jpg \
    --data-shape 384 \
    --thresh 0.5 \
    --nms 0.35 \
    --cpu
