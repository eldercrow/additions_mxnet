python train.py \
    --train-path ./data/VOCdevkit/train.rec \
    --val-path ./data/VOCdevkit/val.rec \
    --num-class 20 \
    --class-names ./dataset/names/pascal_voc.names \
    --network pva101v3 \
    --batch-size 32 \
    --data-shape 384 \
    --optimizer-name nadam \
    --freeze '' \
    --pretrained ./model/ssd_pva101v3_384 \
    --epoch 1000 \
    --lr 1e-05 \
    --use-plateau 1 \
    --lr-steps 4,4 \
    --lr-factor 0.316227766 \
    --end-epoch 300 \
    --frequent 50 \
    --gpus 0,1
