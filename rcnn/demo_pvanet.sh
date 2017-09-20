#!/bin/bash
python demo_pvanet_mpii.py \
    --prefix ./model/pva100_mpii \
    --epoch 41 \
    --gpu 0 \
    --image /home/hyunjoon/faster-rcnn/data/demo/077486027.jpg \
    --vis
    # --prefix /home/hyunjoon/github/additions_mxnet/rcnn/model/pvanet_voc0712 \
