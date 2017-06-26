#!/usr/bin/env bash

# run this experiment with
# nohup bash script/resnet_voc07.sh 0,1 &> resnet_voc07.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to resnet_voc07.log
# gpu=${1:0:1}
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python test.py --network pvanet --gpu 0 --prefix model/pva100_voc0712 --epoch 2
