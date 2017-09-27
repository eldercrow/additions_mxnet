#!/bin/bash
python demo_pvanet_mpii.py \
    --prefix ./model/pva100_mpii \
    --epoch 100 \
    --gpu 4 \
    --image ~/github/additions_mxnet/rcnn/data/mpii/JPEGImages/000003072.jpg \
    --vis
    # --prefix /home/hyunjoon/github/additions_mxnet/rcnn/model/pvanet_voc0712 \
