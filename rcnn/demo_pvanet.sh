#!/bin/bash
python demo_pvanet.py \
    --prefix /home/hyunjoon/github/model_mxnet/pva910/pvanet \
    --epoch 0 \
    --gpu 1 \
    --image /home/hyunjoon/faster-rcnn/data/demo/004545.jpg \
    --vis
