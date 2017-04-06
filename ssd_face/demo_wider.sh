#!/bin/bash
python demo.py \
  --network hjnet_preact \
  --images 1_104 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/hjnet_preact_768 \
  --epoch 23 \
  --data-shape 768 1024 \
  --thresh 0.6 \
  --gpu 1
