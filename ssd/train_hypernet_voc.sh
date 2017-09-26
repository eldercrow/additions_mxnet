python train.py \
    --train-path ./data/VOCdevkit/train.rec \
    --val-path ./data/VOCdevkit/val.rec \
    --num-class 20 \
    --class-names ./dataset/names/pascal_voc.names \
    --network dilatenetv4 \
    --label-width 350 \
    --batch-size 32 \
    --data-shape 384 \
    --optimizer-name nadam \
    --freeze '' \
    --pretrained ~/github/model_mxnet/dilatenet/dilatenetv4_openimage \
    --epoch 1000 \
    --lr 1e-03 \
    --use-plateau 1 \
    --lr-steps 3,3,3,3,4,4,5,5 \
    --lr-factor 0.316227766 \
    --end-epoch 300 \
    --frequent 50 \
    --gpus 0,1
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
