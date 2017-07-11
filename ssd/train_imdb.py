import argparse
# import tools.find_mxnet
import mxnet as mx
import os
import sys
from train.train_net_imdb import train_net

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    parser.add_argument('--dataset', dest='dataset', help='which dataset to use, check .dataset directory', 
                        default='pascal_voc', type=str)
    parser.add_argument('--image-set', dest='image_set', help='train set, can be trainval or train', 
                        default='trainval', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2012',
                        default='2007,2012', type=str)
    parser.add_argument('--val-image-set', dest='val_image_set', 
                        help='validation set, can be val, test or empty', 
                        default='test', type=str)
    parser.add_argument('--val-year', dest='val_year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--devkit-path', dest='devkit_path', help='VOCdevkit path', 
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='pva100_ssd_512',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--from-scratch', dest='from_scratch', type=int, default=0,
                        help='experimental from scratch training')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16_reduced'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=240, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=350,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='150, 200',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=str, default=0.1,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    parser.add_argument('--pattern', dest='monitor_pattern', type=str, default=".*",
                        help='monitor parameter pattern, as regex')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 11-point metric')
    args = parser.parse_args()
    return args

# def parse_class_names(args):
#     """ parse # classes and class_names if applicable """
#     num_class = args.num_class
#     if len(args.class_names) > 0:
#         if os.path.isfile(args.class_names):
#                 # try to open it to read class names
#                 with open(args.class_names, 'r') as f:
#                     class_names = [l.strip() for l in f.readlines()]
#         else:
#             class_names = [c.strip() for c in args.class_names.split(',')]
#         assert len(class_names) == num_class, str(len(class_names))
#         for name in class_names:
#             assert len(name) > 0
#     else:
#         class_names = None
#     return class_names

if __name__ == '__main__':
    # os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # start training
    train_net(args.network, args.dataset, args.image_set, args.devkit_path,
              args.batch_size, args.data_shape, [args.mean_r, args.mean_g, args.mean_b],
              args.resume, args.finetune, args.from_scratch, args.pretrained,
              args.epoch, args.prefix, ctx, args.begin_epoch, args.end_epoch,
              args.frequent, args.learning_rate, args.momentum, args.weight_decay,
              args.lr_refactor_step, args.lr_refactor_ratio,
              year=args.year, 
              val_image_set=args.val_image_set,
              val_year=args.val_year,
              label_pad_width=args.label_width,
              freeze_layer_pattern=args.freeze_pattern,
              iter_monitor=args.monitor,
              monitor_pattern=args.monitor_pattern,
              log_file=args.log_file,
              nms_thresh=args.nms_thresh,
              force_nms=args.force_nms,
              ovp_thresh=args.overlap_thresh,
              use_difficult=args.use_difficult,
              voc07_metric=args.use_voc07_metric)
