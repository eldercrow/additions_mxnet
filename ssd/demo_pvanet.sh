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
  --network pva101_ssd_512 \
  --images $1 \
  --dir image \
  --ext .jpg \
  --prefix model/ssd_512 \
  --epoch 31 \
  --max-data-shapes 640 640 \
  --thresh $TH_POS \
  --gpu 0
