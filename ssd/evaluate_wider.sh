#!/bin/bash

python evaluate_wider.py \
  --dataset wider \
  --image-set val \
  --devkit-path ./data/wider \
  --network hyperface \
  --epoch 1000 \
  --prefix ./model/ssd_sce_hyperface \
  --gpu 1
