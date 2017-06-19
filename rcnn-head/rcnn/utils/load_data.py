import numpy as np
from ..logger import logger
from ..config import config
from ..dataset import *


def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path,
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb, imdb.classes


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb):
    """ remove roidb entries without usable rois """

    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """
        overlaps = entry['max_overlaps']
        fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('load data: filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after))

    return filtered_roidb


def convert_roidb_class(roidb, src_class, dst_class, remove_bg=True):
    ''' convert class idx '''
    if src_class == dst_class:
        return roidb

    if len(src_class) > len(dst_class):
        logger.info('Warning: number of source classes are bigger than that of target classes.')

    src2dst = []
    for scls in src_class:
        if scls not in dst_class:
            logger.info('Warning: {} is not in target, will be converted to background.'.format(scls))
            src2dst.append(0)
        else:
            src2dst.append(dst_class.index(scls))
    src2dst = np.array(src2dst)

    for i, roi_rec in enumerate(roidb):
        # we need to convert gt_classes, gt_overlaps, max_classes
        gt_classes = roi_rec['gt_classes']
        gt_overlaps = np.zeros((gt_classes.size, len(dst_class)), dtype=np.float32)
        gt_classes = src2dst[gt_classes]
        gt_overlap[:, gt_classes] = 1.0

        roi_rec['gt_classes'] = gt_classes
        roi_rec['gt_overlaps'] = gt_overlaps
        roi_rec['max_classes'] = gt_overlaps.argmax(axis=1)

        if remove_bg == True:
            idx = np.where(gt_classes != 0)[0]
            for k, v in roi_rec.items():
                if k == 'flipped': 
                    continue
                if v.ndim == 1:
                    roi_rec[k] = v[idx]
                elif v.ndim == 2:
                    roi_rec[k] = v[idx, :]
                else:
                    assert False
        roidb[i] = roi_rec

    return roidb
