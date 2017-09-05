#!/bin/bash
python demo_face.py \
  --network hyperfacev2 \
  --images ./data/demo/iclr17.jpg \
  --prefix ./model/ssd_hyperfacev2_384 \
  --epoch 1000 \
  --gpu 0 \
  --data-shape 384 \
  --thresh 0.5 \
  # --gpu 0 \
