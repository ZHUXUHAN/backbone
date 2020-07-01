import mxnet as mx


def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type == 'leaky':
        body = mx.sym.LeakyReLU(data=data, act_type='leaky', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    bn_mom    = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type  = kwargs.get('version_act', 'relu')

    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
            no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
            no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), no_bias=True,
            workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')

        return Act(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
            no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
            no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')

        return Act(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    return residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)


def resnet(fp16, units, num_stages, filter_list, num_classes, bottle_neck, **kwargs):
    bn_mom    = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type  = kwargs.get('version_act', 'relu')

    data = mx.sym.Variable(name='data')

    if fp16:
        data = mx.sym.cast(data, dtype='float16')

    body = data
    body = Conv(data=body, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = Act(data=body, act_type=act_type, name='relu0')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], (2, 2), False,
            name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                bottle_neck=bottle_neck, **kwargs)
    if fp16:
        body = mx.sym.cast(body, dtype='float32')

    body = mx.sym.Convolution(data=body, num_filter=int(num_classes / 8), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
        no_bias=True, name="conv_final", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')

    return fc1


def get_symbol(fp16, num_classes, num_layers, **kwargs):
    if num_layers >= 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 122:
        units = [3, 8, 26, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 143:
        units = [3, 8, 33, 3]
    elif num_layers == 146:
        units = [3, 8, 34, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet(
        fp16            = fp16,
        units           = units,
        num_stages      = num_stages,
        filter_list     = filter_list,
        num_classes     = num_classes,
        bottle_neck     = bottle_neck, **kwargs)