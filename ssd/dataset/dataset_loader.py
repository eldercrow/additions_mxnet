import tools.find_mxnet
import mxnet as mx
from pascal_voc import PascalVoc
# from pascal_voc_patch import PascalVocPatch
from wider import Wider
from concat_db import ConcatDB
# from concat_patch_db import ConcatPatchDB


def load_pascal(image_set, year, devkit_path, shuffle=False):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"
    year = [y.strip() for y in year.split(',')]
    assert year, "No year specified"

    # make sure (# sets == # years)
    if len(image_set) > 1 and len(year) == 1:
        year = year * len(image_set)
    if len(image_set) == 1 and len(year) > 1:
        image_set = image_set * len(year)
    assert len(image_set) == len(year), "Number of sets and year mismatch"

    imdbs = []
    for s, y in zip(image_set, year):
        imdbs.append(PascalVoc(s, y, devkit_path, shuffle, is_train=True))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]


def load_wider(image_set, devkit_path, shuffle=False):
    return Wider(image_set, devkit_path, shuffle, is_train=True)


# def load_pascal_patch(image_set, year, devkit_path, patch_shape, shuffle=True):
#     '''
#     '''
#     image_set = [y.strip() for y in image_set.split(',')]
#     assert image_set, "No image_set specified"
#     year = [y.strip() for y in year.split(',')]
#     assert year, "No year specified"
#
#     # make sure (# sets == # years)
#     if len(image_set) > 1 and len(year) == 1:
#         year = year * len(image_set)
#     if len(image_set) == 1 and len(year) > 1:
#         image_set = image_set * len(year)
#     assert len(image_set) == len(year), "Number of sets and year mismatch"
#
#     imdbs = []
#     for s, y in zip(image_set, year):
#         imdbs.append( \
#                 PascalVocPatch(s, y, devkit_path, shuffle=shuffle, is_train=True, patch_shape=patch_shape))
#     if len(imdbs) > 1:
#         return ConcatPatchDB(imdbs, merge_classes=False, shuffle=False)
#     else:
#         return imdbs[0]
#
