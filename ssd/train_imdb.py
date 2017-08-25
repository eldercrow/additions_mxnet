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
    parser.add_argument('--devkit-path', dest='devkit_path', help='VOCdevkit or Wider path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='hypernet',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
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
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=768,
                        help='set image shape')
    parser.add_argument('--force-resize', dest='force_resize', type=bool, default=True,
                        help='resize image to patch size')
    parser.add_argument('--optimizer-name', dest='optimizer_name', type=str, default='adam',
                        help='optimizer name')
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
    parser.add_argument('--use-plateau', dest='use_plateau', type=bool, default=True,
                        help='use plateau learning rate scheduler')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='3,4,5,6',
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
    parser.add_argument('--min-obj-size', dest='min_obj_size', type=float, default=24.0,
                        help='minimum object size to be used for training.')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    args = parser.parse_args()
    return args


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
              args.frequent,
              args.optimizer_name, args.learning_rate, args.momentum, args.weight_decay,
              args.lr_refactor_step, args.lr_refactor_ratio,
              use_plateau=args.use_plateau,
              force_resize=args.force_resize,
              year=args.year,
              freeze_layer_pattern=args.freeze_pattern,
              iter_monitor=args.monitor,
              monitor_pattern=args.monitor_pattern,
              log_file=args.log_file,
              min_obj_size=args.min_obj_size,
              use_difficult=args.use_difficult)
