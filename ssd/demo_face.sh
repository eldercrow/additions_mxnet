#!/bin/bash
python demo_face.py \
  --network fasterface \
  --images ./data/demo/selfie.jpg \
  --prefix ./model/ssd_ff_384 \
  --epoch 1000 \
  --gpu 0 \
  --data-shape 384 \
  --thresh 0.5 \
  # --gpu 0 \
