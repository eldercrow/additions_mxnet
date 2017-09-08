#!/bin/bash
python demo_face.py \
  --network hyperface \
  --images ./data/demo/1_104.jpg \
  --prefix ./model/ssd_hyperface_384 \
  --epoch 1000 \
  --cpu \
  --data-shape 384 \
  --thresh 0.5 \
  # --gpu 0 \
