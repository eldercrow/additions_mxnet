#!/usr/bin/env bash

# run this experiment with
# nohup bash script/resnet_voc07.sh 0,1 &> resnet_voc07.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to resnet_voc07.log
# gpu=${1:0:1}
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python train_end2end.py \
  --network pvanet_mpii \
  --gpu 6 \
  --prefix model/pva100_mpii \
  --dataset mpii \
  --image_set trainval \
  --frequent 500 \
  --lr 1e-03 \
  --lr_step 30,60,90 \
  --pretrained_epoch 0 \
  --pretrained /home/hyunjoon/github/model_mxnet/pva100/pva100 \
  --end_epoch 120
  # --resume \
  # --begin_epoch 41 \
  # --pretrained_epoch 0 \
  # --end_epoch 120
# python test.py --network resnet --gpu 1
