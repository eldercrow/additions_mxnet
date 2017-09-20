"""Presets for various network configurations"""
import logging
import symbol_builder
import numpy as np
from config.config import cfg

def get_config(network, data_shape, **kwargs):
    """Configuration factory for various networks

    Parameters
    ----------
    network : str
        base network name, such as vgg_reduced, inceptionv3, resnet...
    data_shape : int
        input data dimension
    kwargs : dict
        extra arguments
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256, 256]
            strides = [-1, -1, 2, 2, 2, 2, 1]
            pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
                [.75, .8216], [.9, .9721]]
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 512 else [x / 512.0 for x in
                [8, 16, 32, 64, 128, 256, 512]]
        else:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256]
            strides = [-1, -1, 2, 2, 1, 1]
            pads = [-1, -1, 1, 1, 0, 0]
            sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
        if not (data_shape == 300 or data_shape == 512):
            logging.warn('data_shape %d was not tested, use with caucious.' % data_shape)
        return locals()
    elif network == 'inceptionv3':
        from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network in ('hypernet',):
        # network = 'hypernet'
        from_layers = [('hyper{}/1'.format(i), 'hyper{}/2'.format(i)) for i in range(6)]
        num_filters = [-1] * 6
        strides = [-1] * 6
        pads = [-1] * 6
        ratios = [[1, 0.5, 2.0]] * 6
        sizes = [[36, 24], [72, 48], [144, 96], \
                 [288, 192], [data_shape-72, data_shape-48], [data_shape-24, data_shape]]
        sizes = np.array(sizes) / float(data_shape)
        sizes = sizes.tolist()
        sizes[-1] = [sizes[-1][0],]
        normalizations = -1
        steps = []
        th_small = 16.0 / data_shape
        mimic_fc = 2
        python_anchor = True
        return locals()
    elif network in ('hypernetv5', 'hypernetv3', 'hypernetv2', 'hypernetv4', 'hypernetv6'):
        from_layers = [('hyper{}/1'.format(i), 'hyper{}/2'.format(i)) for i in range(6)]
        num_filters = [-1] * 6
        strides = [-1] * 6
        pads = [-1] * 6
        r1 = [1, 2.0, 0.5]
        r2 = [1, np.sqrt(3.0), 1.0 / np.sqrt(3.0), 3.0, 1.0 / 3.0]
        ratios = [r1, r2, r2, r1, r1, r1]
        del r1, r2
        sizes = [[36, 24], [72, 48], [144, 96], \
                 [288, 192], [data_shape-72, data_shape-48], [data_shape-24, data_shape]]
        sizes = np.array(sizes) / float(data_shape)
        sizes = sizes.tolist()
        sizes[-1] = [sizes[-1][0],]
        normalizations = -1
        steps = []
        th_small = 16.0 / data_shape
        mimic_fc = 2 if network == 'hypernetv5' else 1
        dense_vh = True
        python_anchor = True
        return locals()
    elif network in ('hypernetv6',):
        from_layers = [('hyper{}/1'.format(i), 'hyper{}/2'.format(i)) for i in range(6)]
        num_filters = [-1] * 6
        strides = [-1] * 6
        pads = [-1] * 6
        r1 = [1, 2.0, 0.5]
        r2 = [1, np.sqrt(3.0), 1.0 / np.sqrt(3.0), 3.0, 1.0 / 3.0]
        ratios = [r1, r2, r2, r1, r1, r1]
        del r1, r2
        sizes = [[72, 48], [144, 96], \
                 [288, 192], [576, 384], [data_shape-144, data_shape-96], [data_shape-48, data_shape]]
        sizes = np.array(sizes) / float(data_shape)
        sizes = sizes.tolist()
        sizes[-1] = [sizes[-1][0],]
        normalizations = -1
        steps = []
        th_small = 32.0 / data_shape
        mimic_fc = 1
        dense_vh = True
        python_anchor = True
        return locals()
    elif network == 'pva101':
        # network = 'pva101'
        assert data_shape == 384
        from_layers = ['hyper2/relu', 'hyper3/relu', 'hyper4/relu', '', '', '']
        num_filters = [-1, -1, -1, 512, 256, 256]
        strides = [-1, -1, -1, 2, 2, 2]
        pads = [-1, -1, -1, 1, 1, 1]
        r1 = [1, np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        r2 = [1, np.sqrt(3.0), 1.0 / np.sqrt(3.0), 3.0, 1.0 / 3.0]
        ratios = [r1, r2, r2, r2, r1, r1]
        del r1, r2
        sizes = [[32, 24], [64, 48], [128, 96], \
                 [256, 192], [data_shape-64, data_shape-48], [data_shape-32, data_shape]]
        sizes = np.array(sizes) / float(data_shape)
        sizes = sizes.tolist()
        normalizations = -1
        steps = []
        th_small = 16.0 / data_shape
        mimic_fc = 2
        python_anchor = True
        return locals()
    elif network in ('pva101v2', 'pva101v3', 'pva101v4'):
        # network = 'pva101'
        assert data_shape == 384
        from_layers = [('hyper{}_0/relu'.format(i), 'hyper{}_1/relu'.format(i)) for i in range(6)]
        num_filters = [-1] * 6
        strides = [-1] * 6
        pads = [-1] * 6
        r1 = [1, 2.0, 0.5]
        r2 = [1, np.sqrt(3.0), 1.0 / np.sqrt(3.0), 3.0, 1.0 / 3.0]
        ratios = [r1, r2, r2, r1, r1, r1]
        del r1, r2
        sizes = [[36, 24], [72, 48], [144, 96], \
                 [288, 192], [data_shape-72, data_shape-48], [data_shape-24, data_shape]]
        sizes = np.array(sizes) / float(data_shape)
        sizes = sizes.tolist()
        sizes[-1] = [sizes[-1][0],]
        normalizations = -1
        steps = []
        th_small = 16.0 / data_shape
        mimic_fc = 1
        dense_vh = True
        python_anchor = True
        return locals()
    elif network == 'ssd_pva':
        # network = 'pva101'
        assert data_shape == 512
        from_layers = ['hyper{}/relu'.format(i) for i in range(6)]
        num_filters = [-1] * 6
        strides = [-1] * 6
        pads = [-1] * 6
        r1 = [1, np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        r2 = [1, np.sqrt(3.0), 1.0 / np.sqrt(3.0), 3.0, 1.0 / 3.0]
        ratios = [r1, r2, r2, r2, r1, r1]
        del r1, r2
        sizes = [[32, 24], [64, 48], [128, 96], \
                 [256, 192], [data_shape-64, data_shape-48], [data_shape-32, data_shape]]
        sizes = np.array(sizes) / float(data_shape)
        sizes = sizes.tolist()
        normalizations = -1
        steps = []
        th_small = 18.0 / data_shape
        return locals()
    elif network in ('facenet',):
        # network = 'facenet'
        sz_list = []
        sz0 = 12.0
        sz_ratio = np.power(2.0, 0.5)
        for _ in range(6):
            sz_list.append(sz0)
            sz0 *= 2
        from_layers = [('hyper{}/1'.format(i), 'hyper{}/2'.format(i)) for i in range(6)]
        num_filters = [-1] * len(from_layers)
        strides = [-1] * len(from_layers)
        pads = [-1] * len(from_layers)
        ratios = [[0.8,]] * len(from_layers)
        sizes = [[s, s * sz_ratio] for s in sz_list]
        sizes[-1] = [s,]
        normalizations = -1
        steps = [2**(2+i) for i in range(len(sz_list))]
        th_small = 8.0
        mimic_fc = 1
        python_anchor = True
        del sz_list, sz0, sz_ratio
        return locals()
    elif network in ('fasterface', 'hyperface', 'hyperfacev2', 'hyperfacev3', 'dilatefacev1'):
        # network = 'facenet'
        sz_list = []
        sz0 = 12.0
        sz_ratio = np.power(2.0, 0.333333)
        for _ in range(6):
            sz_list.append(sz0)
            sz0 *= 2
        from_layers = [('hyper{}/1'.format(i), 'hyper{}/2'.format(i)) for i in range(len(sz_list))]
        num_filters = [-1] * len(from_layers)
        strides = [-1] * len(from_layers)
        pads = [-1] * len(from_layers)
        rr = [[1.0,]]
        ratios = rr * len(from_layers)
        sizes = [[ss / sz_ratio, ss, ss * sz_ratio] for ss in sz_list]
        # sizes[0] = [12.0, 12.0 * sz_ratio]
        sizes[-1] = [sz_list[-1] / sz_ratio, sz_list[-1]]
        normalizations = -1
        upscales = 1
        steps = [2**(2+i) for i in range(len(sz_list))]
        th_small = 6.0
        python_anchor = True
        dense_vh = True
        square_bb = True
        # mimic_fc = 1
        del sz_list, sz0, sz_ratio, rr
        return locals()
    # elif network in ('hyperfacev3',):
    #     # network = 'facenet'
    #     sz_list = []
    #     sz0 = 12.0
    #     sz_ratio = np.power(2.0, 0.333333)
    #     for _ in range(7):
    #         sz_list.append(sz0)
    #         sz0 *= 2
    #     from_layers = [('hyper{}/1'.format(i), 'hyper{}/2'.format(i)) for i in range(len(sz_list))]
    #     num_filters = [-1] * len(from_layers)
    #     strides = [-1] * len(from_layers)
    #     pads = [-1] * len(from_layers)
    #     ratios = [[1.0,]] * len(from_layers)
    #     sizes = [[ss / sz_ratio, ss, ss * sz_ratio] for ss in sz_list]
    #     # sizes[0] = [12.0, 12.0 * sz_ratio]
    #     sizes[-1] = [sz_list[-1] / sz_ratio, sz_list[-1]]
    #     normalizations = -1
    #     upscales = 1
    #     steps = [2**(2+i) for i in range(len(sz_list))]
    #     th_small = 8.0
    #     python_anchor = True
    #     square_bb = True
    #     # mimic_fc = 1
    #     del sz_list, sz0, sz_ratio
    #     return locals()
    else:
        msg = 'No configuration found for %s with data_shape %d' % (network, data_shape)
        raise NotImplementedError(msg)

def get_symbol_train(network, data_shape, **kwargs):
    """Wrapper for get symbol for train

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol_train for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol_train(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder.get_symbol_train(**config)

def get_symbol(network, data_shape, **kwargs):
    """Wrapper for get symbol for test

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder.get_symbol(**config)
