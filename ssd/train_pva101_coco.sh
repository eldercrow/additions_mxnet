python train.py \
    --train-path ./data/coco/train.rec \
    --val-path ./data/coco/val.rec \
    --num-class 80 \
    --class-names ./dataset/names/mscoco.names \
    --label-width 560 \
    --network pva101v2 \
    --batch-size 32 \
    --data-shape 384 \
    --optimizer-name nadam \
    --freeze '' \
    --resume 3 \
    --lr 1e-03 \
    --use-plateau 0 \
    --lr-steps 30,30,20,20,10,10 \
    --lr-factor 0.316227766 \
    --end-epoch 300 \
    --frequent 100 \
    --gpus 4,5
    # --pretrained ~/github/model_mxnet/pva100/pva100 \
    # --epoch 0 \
