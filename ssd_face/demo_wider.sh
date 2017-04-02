#!/bin/bash
python demo.py \
  --network hjnet_preact \
  --images 1_20 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/hjnet_preact \
  --epoch 5 \
  --data-shape 768 \
  --cpu
