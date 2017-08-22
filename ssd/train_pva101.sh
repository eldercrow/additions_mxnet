python train.py \
    --network pva101 \
    --batch-size 32 \
    --data-shape 384 \
    --optimizer-name sgd \
    --freeze '' \
    --resume 63 \
    --lr 1e-03 \
    --lr-steps 4,4,6,6,8 \
    --lr-factor 0.316227766 \
    --end-epoch 240 \
    --frequent 50 \
    --gpus 0
    # --pretrained /home/hyunjoon/github/model_mxnet/pva100/pva100 \
    # --epoch 0 \
