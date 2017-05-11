#!/bin/bash
python demo.py \
  --network spotnet_lite3_bnfixed \
  --images $1 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/spotnet_lite3_768 \
  --epoch 52 \
  --max-data-shapes 2560 2560 \
  --thresh 0.5 \
  --gpu 0
