#!/bin/bash
python demo_face.py \
  --network facenet \
  --images ./data/demo/1_156.jpg \
  --prefix ./model/ssd_facenet_480 \
  --epoch 1000 \
  --gpu 0 \
  --data-shape 480 \
  --thresh 0.55 \
  # --gpu 0 \
