import os
from utils import DotDict, namedtuple_with_defaults, zip_namedtuple, config_as_dict

RandScaler = namedtuple_with_defaults('RandScaler',
    'min_aug_scale, max_aug_scale, min_aug_gt_scale, max_aug_trials, max_aug_sample, aug_patch_size', 
    [0.5, 1.5, 32.0 / 512.0, 50, 1, 512])

RandCropper = namedtuple_with_defaults('RandCropper',
    'min_crop_scales, max_crop_scales, \
    min_crop_aspect_ratios, max_crop_aspect_ratios, \
    min_crop_overlaps, max_crop_overlaps, \
    min_crop_sample_coverages, max_crop_sample_coverages, \
    min_crop_object_coverages, max_crop_object_coverages, \
    max_crop_trials',
    [0.0, 1.0,
    0.5, 2.0,
    0.0, 1.0,
    0.0, 1.0,
    0.0, 1.0,
    25])

RandPadder = namedtuple_with_defaults('RandPadder',
    'rand_pad_prob, max_pad_scale, fill_value',
    [0.0, 1.0, 127])

ColorJitter = namedtuple_with_defaults('ColorJitter',
    'random_hue_prob, max_random_hue, \
    random_saturation_prob, max_random_saturation, \
    random_illumination_prob, max_random_illumination, \
    random_contrast_prob, max_random_contrast',
    [0.0, 18,
    0.0, 32,
    0.0, 32,
    0.0, 0.5])


cfg = DotDict()
cfg.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# training configs
cfg.train = DotDict()
# random scaling/cropping samplers
cfg.train.rand_scaler = RandScaler(min_aug_scale=0.5, max_aug_scale=1.5, aug_patch_size=512)
# random cropping samplers
cfg.train.rand_crop_samplers = [
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.1),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.3),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.5),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.7),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.9),]
cfg.train.crop_emit_mode = 'center'
# cfg.train.emit_overlap_thresh = 0.4
# random padding
cfg.train.rand_pad = RandPadder(rand_pad_prob=0.5, max_pad_scale=4.0)
# random color jitter
cfg.train.color_jitter = ColorJitter(random_hue_prob=0.5, random_saturation_prob=0.5,
    random_illumination_prob=0.5, random_contrast_prob=0.5)
cfg.train.inter_method = 10  # random interpolation
cfg.train.rand_mirror_prob = 0.5
cfg.train.shuffle = True
cfg.train.seed = 233
cfg.train.preprocess_threads = 6
# for imdb based training
cfg.train.rand_mirror = True
cfg.train.init_shuffle = True
cfg.train.epoch_shuffle = True
cfg.train.resize_epoch = 1
cfg.train = config_as_dict(cfg.train)  # convert to normal dict

# validation
cfg.valid = DotDict()
cfg.valid.rand_scaler = RandScaler(min_aug_scale=0.5, max_aug_scale=1.5, aug_patch_size=512)
cfg.valid.rand_crop_samplers = []
cfg.valid.rand_pad = RandPadder()
cfg.valid.color_jitter = ColorJitter()
cfg.valid.rand_mirror_prob = 0
cfg.valid.shuffle = False
cfg.valid.seed = 0
# for imdb based training
cfg.valid.rand_mirror = True
cfg.valid.init_shuffle = True
cfg.valid.epoch_shuffle = True
cfg.valid = config_as_dict(cfg.valid)  # convert to normal dict
