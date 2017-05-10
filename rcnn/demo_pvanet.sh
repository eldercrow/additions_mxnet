#!/bin/bash
python demo_pvanet.py \
    --prefix /home/hyunjoon/github/additions_mxnet/rcnn/model/pva911_twn_voc0712 \
    --epoch 32 \
    --gpu 0 \
    --image /home/hyunjoon/faster-rcnn/data/demo/test_comp.jpg \
    --vis
    # --prefix /home/hyunjoon/github/additions_mxnet/rcnn/model/pvanet_voc0712 \
