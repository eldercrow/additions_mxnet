#!/bin/bash
python demo_pvanet.py \
    --prefix ./model/pva100_voc0712 \
    --epoch 75 \
    --gpu 0 \
    --image /home/hyunjoon/faster-rcnn/data/demo/001150.jpg \
    --vis
    # --prefix /home/hyunjoon/github/additions_mxnet/rcnn/model/pvanet_voc0712 \
