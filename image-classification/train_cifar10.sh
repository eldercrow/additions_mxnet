#!/bin/bash
python train_cifar10.py --network inception-bn-twn \
                        --optimizer adam \
                        --lr 0.002 \
                        --wd 0.0001 \
                        --gpus 2 \
                        --monitor 5000 \
                        --disp-batches 500 \
                        --model-prefix ./model/cifar10-inception-bn-twn \
                        --pretrained ./model/cifar10-inception-bn \
                        --pretrained-epoch 172
                        
