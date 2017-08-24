#!/bin/bash

python evaluate_wider.py \
  --dataset wider \
  --image-set val \
  --devkit-path ./data/wider \
  --network fasterface \
  --epoch 1000 \
  --prefix ./model/ssd_ff \
  --cpu
