#!/usr/bin/bash

python score.py \
  --model ./model/mobilenetv6_imagenet,129 \
  --data-val ./data/imagenet/imagenet1k-val.rec \
  --rgb-mean 123.68,116.779,103.939 \
  --image-shape 3,224,224 \
  --gpus 0,1
