import mxnet as mx
from symbols.net_block_sep import *


def inception(data, name, f1, f3, f5, do_pool, use_global_stats):
    #
    pool1 = pool(data, name=name+'pool1/') if do_pool else data

    kernel = (4, 4) if do_pool else (3, 3)
    stride = (2, 2) if do_pool else (1, 1)

    # conv1
    conv1 = relu_conv_bn(pool1, name+'conv1/',
            num_filter=f1, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    # conv3
    conv3_1 = relu_conv_bn(data, name+'conv3_1/',
            num_filter=f3[0], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    conv3_2 = depthwise_conv(conv3_1, name+'conv3_2/',
            nf_dw=f3[0], nf_sep=f3[1], kernel=kernel, stride=stride,
            use_global_stats=use_global_stats)

    # conv5
    conv5_1 = relu_conv_bn(data, name+'conv5_1/',
            num_filter=f5[0], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    conv5_2 = depthwise_conv(conv5_1, name+'conv5_2/',
            nf_dw=f5[0], nf_sep=f5[1], use_global_stats=use_global_stats)
    conv5_3 = depthwise_conv(conv5_2, name+'conv5_3/',
            nf_dw=f5[1], nf_sep=f5[2], kernel=kernel, stride=stride,
            use_global_stats=use_global_stats)

    return mx.sym.concat(conv1, conv3_2, conv5_3)


def proj_add(lhs, rhs, name, num_filter, do_pool, use_global_stats):
    #
    lhs = pool(lhs) if do_pool else lhs
    lhs = relu_conv_bn(lhs, name+'lhs/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    rhs = relu_conv_bn(rhs, name+'rhs/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    return mx.sym.broadcast_add(lhs, rhs, name=name+'add/')


def topdown_feature(data, updata, name, scale, nf_proj, nf_all, nf_sqz, use_global_stats, n_mix_iter=0):
    #
    # upsample, proj, concat, mix
    updata = mx.sym.UpSampling(updata, scale=scale, sample_type='bilinear',
            num_filter=nf_all, name=name+'upsample')

    data = mx.sym.concat(data, updata, name=name+'concat')
    data = relu_conv_bn(data, name+'proj/',
            num_filter=nf_all, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    data = depthwise_conv(data, name+'mix/',
            nf_dw=nf_all, nf_sep=nf_all, use_global_stats=use_global_stats)

    for i in range(n_mix_iter):
        d0 = data
        data = relu_conv_bn(data, name+'mix1x1/{}/'.format(i),
                num_filter=nf_sqz, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        data = depthwise_conv(data, name+'mix3x3/{}/'.format(i),
                nf_dw=nf_sqz, nf_sep=nf_all, use_global_stats=use_global_stats)
        data = data + d0
    return data


def prepare_groups(data, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3 1
    n_units = [4, 4, 2]
    filters_1 = [192, 256, 256]
    filters_3 = [(64, 128), (96, 192), (96, 192)]
    filters_5 = [(64, 64, 64) for _ in range(4)]
    filters_a = (384, 512, 512)

    groups = [data]
    g = data
    for i, (nu, f1, f3, f5, fa) in enumerate(zip(n_units, filters_1, filters_3, filters_5, filters_a)):
        g0 = g
        for j in range(nu):
            name = 'inc{}/{}/'.format(i+3, j+1)
            g = inception(g, name, f1=f1, f3=f3, f5=f5, do_pool=(j==0),
                    use_global_stats=use_global_stats)
            if (j+1) % 2 == 0:
                g = proj_add(g0, g, 'res{}/{}/'.format(i+3, j+1),
                        num_filter=fa, do_pool=(j==1), use_global_stats=use_global_stats)
                g0 = g

        groups.append(g)

    g = depthwise_conv(g, 'g4/1/'.format(i),
            nf_dw=512, nf_sep=512, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    g = depthwise_conv(g, 'g4/2/'.format(i),
            nf_dw=512, nf_sep=512, use_global_stats=use_global_stats)
    groups.append(g)

    g = depthwise_conv(g, 'g5/1/'.format(i),
            nf_dw=512, nf_sep=512, pad=(0, 0), use_global_stats=use_global_stats)
    groups.append(g)

    # top-down features
    groups[1] = topdown_feature(groups[1], groups[2], 'up1/', scale=2,
            nf_proj=128, nf_all=512, nf_sqz=256, use_global_stats=use_global_stats)
    groups[0] = topdown_feature(groups[0], groups[1], 'up0/', scale=2,
            nf_proj=256, nf_all=512, nf_sqz=256, use_global_stats=use_global_stats)

    return groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    # label = mx.symbol.Variable(name="label")

    conv1 = mx.sym.Convolution(data, name='1/conv',
            num_filter=16, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)

    bn2 = depthwise_conv(bn1, '2_1/',
            nf_dw=32, nf_sep=64,use_global_stats=use_global_stats)
    bn2 = depthwise_conv(bn2, '2_2/',
            nf_dw=64, nf_sep=64, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)

    bn3 = depthwise_conv(bn2, '3_1/',
            nf_dw=64, nf_sep=128, use_global_stats=use_global_stats)
    # bn3 = depthwise_conv(bn3, '3_2/',
    #         nf_dw=128, nf_sep=128, use_global_stats=use_global_stats)
    bn3 = depthwise_conv(bn3, '3_3/',
            nf_dw=128, nf_sep=128, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)

    bn4 = depthwise_conv(bn3, '4_1/',
            nf_dw=128, nf_sep=192, use_global_stats=use_global_stats)
    # bn4 = depthwise_conv(bn4, '4_2/',
    #         nf_dw=192, nf_sep=192, use_global_stats=use_global_stats)
    bn4 = depthwise_conv(bn4, '4_3/',
            nf_dw=192, nf_sep=256, use_global_stats=use_global_stats)
    bn4 = depthwise_conv(bn4, '4_4/',
            nf_dw=256, nf_sep=256, use_global_stats=use_global_stats)

    groups = prepare_groups(bn4, use_global_stats)

    g3 = groups[3]
    g3 = relu_conv_bn(g3, '5/',
            num_filter=2048, kernel=(3, 3), pad=(0, 0),
            use_global_stats=use_global_stats)
    # g3 = depthwise_conv(g3, '5/',
    #         nf_dw=512, nf_sep=1024, kernel=(3, 3), pad=(0, 0), stride=(2, 2),
    #         use_global_stats=use_global_stats)

    pool6 = mx.sym.Activation(g3, act_type='relu')
    # pool6 = mx.sym.Pooling(g3, name='pool6',kernel=(1, 1), global_pool=True, pool_type='avg')
    fc7 = mx.sym.Convolution(pool6, name='fc7',
            num_filter=num_classes, kernel=(1, 1), pad=(0, 0), no_bias=False)
    flatten = mx.sym.Flatten(fc7, name='flatten')

    softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    return softmax
