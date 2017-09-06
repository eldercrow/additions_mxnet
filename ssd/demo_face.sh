#!/bin/bash
python demo_face.py \
  --network hyperfacev3 \
  --images ./data/demo/1_20.jpg \
  --prefix ./model/ssd_hyperfacev3_768 \
  --epoch 1000 \
  --cpu \
  --data-shape 768 \
  --thresh 0.5 \
  # --gpu 0 \
