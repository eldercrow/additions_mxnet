#!/bin/bash
python demo_pvanet.py \
    --prefix /home/hyunjoon/github/model_mxnet/pva910/pvanet \
    --epoch 0 \
    --gpu 0 \
    --image /home/hyunjoon/faster-rcnn/data/demo/001763.jpg \
    --vis
    # --prefix /home/hyunjoon/github/additions_mxnet/rcnn/model/pvanet_voc0712 \
