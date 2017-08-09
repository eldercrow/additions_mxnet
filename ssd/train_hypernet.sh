python train.py \
    --network hypernet \
    --batch-size 16 \
    --data-shape 448 \
    --optimizer-name sgd \
    --freeze '' \
    --pretrained none \
    --epoch 1000 \
    --lr 0.002 \
    --frequent 100 \
    --gpus 0,1