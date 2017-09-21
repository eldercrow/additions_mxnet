#!/bin/bash
python demo_face.py \
  --network dilatefacev2 \
  --images ./data/demo/1_104.jpg \
  --prefix ./model/ssd_dilatefacev2_384 \
  --epoch 1000 \
  --cpu \
  --data-shape 384 \
  --thresh 0.35 \
  # --gpu 0 \
