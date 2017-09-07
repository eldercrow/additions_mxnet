#!/bin/bash
python demo_face.py \
  --network hyperface \
  --images ./data/demo/1_78.jpg \
  --prefix ./model/ssd_sce04_hyperface_384 \
  --epoch 1000 \
  --gpu 0 \
  --data-shape 384 \
  --thresh 0.5 \
  # --gpu 0 \
