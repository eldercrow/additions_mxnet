#!/bin/bash

python demo.py \
    --network pva101v2 \
    --prefix ./model/ssd_pva101v2_384 \
    --epoch 1000 \
    --images ./data/demo/004545.jpg \
    --data-shape 384 \
    --thresh 0.5 \
    --nms 0.45 \
    --cpu
