#!/bin/bash
python demo_face.py \
  --network hyperface \
  --images ./data/demo/1_20.jpg \
  --prefix ./model/ssd_hyperface_hyperface_384 \
  --epoch 1000 \
  --gpu 0 \
  --data-shape 384 \
  --thresh 0.25 \
  # --gpu 0 \
