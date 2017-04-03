#!/bin/bash
python demo.py \
  --network hjnet_preact \
  --images 1_465 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/hjnet_preact_768 \
  --epoch 21 \
  --data-shape 1024 768 \
  --gpu 1
