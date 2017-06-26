#!/bin/bash
python demo_pvanet.py \
    --prefix model/pva100_voc0712 \
    --epoch 33 \
    --gpu 1 \
    --image /home/hyunjoon/faster-rcnn/data/demo/004545.jpg \
    --vis
    # --prefix /home/hyunjoon/github/additions_mxnet/rcnn/model/pvanet_voc0712 \
