#!/bin/bash
python demo.py \
  --network pvtnet_preact \
  --images 2_58 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/pvtnet_preact_768 \
  --epoch 9 \
  --data-shape 1024 768 \
  --thresh 0.5 \
  --cpu
