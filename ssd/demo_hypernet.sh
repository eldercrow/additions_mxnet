#!/bin/bash

python demo.py \
    --network hypernet \
    --prefix ./model/ssd_hypernet_448 \
    --epoch 1000 \
    --data-shape 448 \
    --gpu 0
