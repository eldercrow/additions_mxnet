#!/bin/bash
python demo_face.py \
  --network fasterface \
  --images ./data/demo/1_20.jpg \
  --prefix ./model/ssd_ff_384 \
  --epoch 1000 \
  --cpu \
  --data-shape 384 \
  --thresh 0.55 \
  # --gpu 0 \
