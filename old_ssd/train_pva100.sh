python train_imdb.py \
    --gpus 1 \
    --dataset pascal_voc \
    --year 2007,2012 \
    --network pva102_ssd_512 \
    --prefix model/pva102_ssd \
    --freeze '' \
    --batch-size 16 \
    --data-shape 384 \
    --val-image-set '' \
    --lr 1e-03 \
    --lr-factor 0.316227766 \
    --lr-steps 2,2,3,3,4,4,4 \
    --frequent 100 \
    --wd 1e-04 \
    --pretrained ./model/pva101 \
    --epoch 0
    # --resume 24
    # --pretrained ./model/pva102_ssd_512 \
