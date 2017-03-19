#!/bin/bash
python train_imagenet.py --network hjnet_preact \
                               --optimizer adam \
                               --lr 0.001 \
                               --wd 0.0001 \
                               --top-k 5 \
                               --data-train /home/hyunjoon/datasets/ILSVRC2012_cls/ILSVRC2012_train.rec \
                               --gpus 0 \
                               --batch-size 48 \
                               --monitor 5000 \
                               --disp-batches 100 \
                               --model-prefix ./model/imagenet1k-hjnet_preact \
                               --image-shape 3,200,200
                               # --batch-per-epoch 5000 \
                               # --lr-step-epochs 30,45,60,70 \
                               # --lr-factor 0.316227766 \

# python hjnet_train_imagenet.py --network hjnet_student \
#                                --model-prefix ./model/imagenet1k-hjnet_student\
#                                --load-epoch 0 \
#                                --optimizer sgd \
#                                --top-k 5 \
#                                --data-train /home/hyunjoon/datasets/ILSVRC2012_cls/ILSVRC2012_train.rec \
#                                --gpus 1 \
#                                --batch-size 16
