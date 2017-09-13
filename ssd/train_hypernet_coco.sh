python train.py \
    --train-path ./data/coco/train.rec \
    --val-path ./data/coco/val.rec \
    --num-class 80 \
    --class-names ./dataset/names/mscoco.names \
    --network hypernetv3 \
    --label-width 560 \
    --batch-size 32 \
    --data-shape 384 \
    --optimizer-name sgd \
    --freeze '' \
    --resume 23 \
    --lr 1e-02 \
    --use-plateau 1 \
    --lr-steps 3,3,4,4,5,5,6 \
    --lr-factor 0.316227766 \
    --end-epoch 300 \
    --frequent 100 \
    --gpus 0,1
    # --pretrained ./model/ssd_hypernetv3_voc_384 \
    # --epoch 0 \

# python train_imdb.py \
#     --network hypernetv5 \
#     --dataset pascal_voc \
#     --devkit-path ./data/VOCdevkit \
#     --year 2007,2012 \
#     --image-set trainval \
#     --val-image-set test \
#     --val-year 2007 \
#     --batch-size 32 \
#     --data-shape 384 \
#     --optimizer-name sgd \
#     --pretrained ./model/ssd_hypernetv5_384 \
#     --epoch 1000 \
#     --freeze '' \
#     --lr 1e-02 \
#     --use-plateau 1 \
#     --lr-factor 0.316227766 \
#     --lr-steps 2,3,3,4,4,5,5 \
#     --end-epoch 250 \
#     --frequent 100 \
#     --gpus 0,1
#     # --resume 18 \
