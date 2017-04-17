#!/bin/bash
python demo.py \
  --network phgnet \
  --images 1_20 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/phgnet_768 \
  --epoch 20 \
  --data-shape 768 1024 \
  --thresh 0.7 \
  --gpu 1
