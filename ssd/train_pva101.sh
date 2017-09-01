python train.py \
    --network pva101v2 \
    --batch-size 16 \
    --data-shape 384 \
    --optimizer-name nadam \
    --freeze '' \
    --pretrained /home/hyunjoon/github/model_mxnet/pva100/pva100 \
    --epoch 0 \
    --lr 1e-03 \
    --use-plateau 1 \
    --lr-steps 2,3,3,4,4,3,3 \
    --lr-factor 0.316227766 \
    --end-epoch 300 \
    --frequent 50 \
    --gpus 0
