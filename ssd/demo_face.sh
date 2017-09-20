#!/bin/bash
python demo_face.py \
  --network hyperfacev3 \
  --images ./data/demo/1_78.jpg \
  --prefix ./model/ssd_hyperfacev3_384 \
  --epoch 1000 \
  --cpu \
  --data-shape 384 \
  --thresh 0.35 \
  # --gpu 0 \
