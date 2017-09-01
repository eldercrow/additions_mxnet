#!/bin/bash

python evaluate.py \
    --network pva101v2 \
    --data-shape 384 \
    --prefix ./model/ssd_pva101v2 \
    --epoch 1000 \
    --gpus 0 
