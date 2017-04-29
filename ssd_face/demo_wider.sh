#!/bin/bash
python demo.py \
  --network spotnet_xy \
  --images $1 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/spotnet_xy_768 \
  --epoch 40 \
  --data-shape 768 1024 \
  --thresh 0.8 \
  --gpu 0
