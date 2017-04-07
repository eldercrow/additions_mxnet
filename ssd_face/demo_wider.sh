#!/bin/bash
python demo.py \
  --network phgnet \
  --images 2_58 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/phgnet_768 \
  --epoch 5 \
  --data-shape 768 1024 \
  --thresh 0.6 \
  --gpu 0
