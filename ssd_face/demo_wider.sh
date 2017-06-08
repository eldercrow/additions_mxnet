#!/bin/bash
if [[ "$#" -lt 1 ]]; then
  echo "Image name not given."
  exit
fi
if [[ "$#" -gt 1 ]] 
then
  TH_POS=$2
else
  TH_POS=0.55
fi

python demo.py \
  --network spotnet_lighter \
  --images $1 \
  --dir image \
  --ext .jpg \
  --prefix model/spotnet_lighter_trainval_bnfixed_768 \
  --epoch 34 \
  --max-data-shapes 3072 3072 \
  --thresh $TH_POS \
  --gpu 0
  # --cpu 
