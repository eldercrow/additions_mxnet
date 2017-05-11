#!/bin/bash
if [[ "$#" -lt 1 ]]; then
  echo "Image name not given."
  exit
fi
if [[ "$#" -gt 1 ]] 
then
  TH_POS=$2
else
  TH_POS=0.5
fi

python demo.py \
  --network spotnet_lite3_bnfixed \
  --images $1 \
  --dir image \
  --ext .jpg \
  --prefix /home/hyunjoon/github/additions_mxnet/ssd_face/model/spotnet_lite3_768 \
  --epoch 52 \
  --max-data-shapes 2560 2560 \
  --thresh $TH_POS \
  --gpu 0
