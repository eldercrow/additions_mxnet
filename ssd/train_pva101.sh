python train.py \
    --network pva101 \
    --batch-size 32 \
    --data-shape 384 \
    --optimizer-name sgd \
    --freeze '' \
    --pretrained /home/hyunjoon/github/model_mxnet/pva100/pva100 \
    --epoch 0 \
    --lr 1e-02 \
    --lr-steps 40,40,40,40,20,20,10,10 \
    --lr-factor 0.316227766 \
    --end-epoch 240 \
    --frequent 50 \
    --gpus 0
