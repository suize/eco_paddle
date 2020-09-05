import yaml
import paddle

import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

import paddle.fluid.dygraph as dygraph




class ECOfull(fluid.dygraph.Layer):
    def __init__(self, model_path='tf_model_zoo/ECOfull/ECOfull.yaml', num_classes=101,
                       num_segments=32, pretrained_parts='both'):

        super(ECOfull, self).__init__()

        self.num_segments = num_segments

        self.pretrained_parts = pretrained_parts

        manifest = yaml.load(open(model_path))

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        self.new_fc = fluid.dygraph.Linear(400,101,act='softmax')
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            # print('op:',op)

            # if op == 'ReLU' or op == 'Pooling3d':
            #     self.
            #     pass
            if op != 'Concat' and op != 'Eltwise':
                
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True if op == 'Conv3d' else True, num_segments=num_segments)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            elif op == 'Concat':
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = self._channel_dict[in_var[0]]
                self._channel_dict[out_var[0]] = channel


    def forward(self, input,label=None):
        # import torch
        data_dict = dict()
        # data_dict[self._op_list[0][-1]] = input
        sample_len = 3
        # print('*'*100)
        # print(input.shape)
        input = fluid.layers.reshape(input,shape=[-1, sample_len] + input.shape[-2:])
        # print(self._op_list[0])
        data_dict[self._op_list[0][-1]] = input

        for op in self._op_list:
            
            # print('-'*100)
            # print(op)
            # print(op[2].shape)
            # print('-'*100)
            if op[1] == 'ReLU':
                data_dict[op[2]] = fluid.layers.relu(data_dict[op[-1]])
                
            elif op[1] == 'Pooling3d':
                """
                {'kernel_d': 'N/4',
                'kernel_h': 7,
                'kernel_w': 7,
                'stride': 1,
                'mode': 'ave'}
                """

                
                if op[0] == 'global_pool3D':
                    b,c,d,h,w = data_dict[op[-1]].shape
                    data_dict[op[2]] = fluid.layers.pool3d(data_dict[op[-1]],pool_size=[int(d),7,7],pool_stride=1,pool_type='avg')
                else:
                    # print('1'*100)
                    layer_output = data_dict[op[-1]]
                    # print(layer_output.shape)
                    layer_transpose_output = fluid.layers.transpose(fluid.layers.reshape(layer_output,[-1, self.num_segments] + layer_output.shape[1:]), [0,2,1,3,4])

                    # print(layer_transpose_output.shape)
                    # print('2'*100)

                    b,c,d,h,w = layer_transpose_output.shape
                    
                    y = layer_transpose_output#fluid.layers.reshape(layer_transpose_output,[b,1,d,h,w])
                    data_dict[op[2]] = fluid.layers.pool3d(y,pool_size=[d,1,1],pool_stride=1,pool_type='avg')
                    b,c,d,h,w = data_dict[op[2]].shape
                    # data_dict[op[2]] = fluid.layers.reshape(data_dict[op[2]],[b,c,h,w])

            elif op[1] != 'Concat' and op[1] != 'InnerProduct' and op[1] != 'Eltwise':
                # first 3d conv layer judge, the last 2d conv layer's output must be transpose from 4d to 5d
                if op[0] == 'res3a_2' or op[0] == 'global_pool2D_reshape_consensus':
                    layer_output = data_dict[op[-1]]
                    layer_transpose_output = fluid.layers.transpose(fluid.layers.reshape(layer_output,[-1, self.num_segments] + layer_output.shape[1:]), [0,2,1,3,4])
                    data_dict[op[2]] = getattr(self, op[0])(layer_transpose_output)
                else:
                    data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                    # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(fluid.layers.reshape(x,[x.shape[0],-1]))
                # (x.view(x.size(0), -1))
            elif op[1] == 'Eltwise':
                try:
                    data_dict[op[2]] = fluid.layers.elementwise_add(data_dict[op[-1][0]], data_dict[op[-1][1]],1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].shape)
                    raise
                # x = data_dict[op[-1]]
                # data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = fluid.layers.concat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].shape)
                    raise
            

            # print('-'*100)
            # print(data_dict[op[2]].shape)
            # print(op)
            # print('-'*100)
    # print output data size in each layers
        # for k in data_dict.keys():
        #     print(k,": ",data_dict[k].size())
        # exit()

        # "self._op_list[-1][2]" represents: last layer's name(e.g. fc_action)

        x = data_dict[self._op_list[-1][2]]
        # x = fluid.layers.reduce_mean(x,dim=1)

        # x = fluid.layers.reshape(x,[x.shape[0],x.shape[-1]])
        # print('x_reduce_mean_shape:',x.shape)
        x = self.new_fc(x)
        if label is not None:
            acc = fluid.layers.accuracy(input = x,label = label)
            return x,acc

        return x





LAYER_BUILDER_DICT=dict()


def parse_expr(expr):
    parts = expr.split('<=')
    return parts[0].split(','), parts[1], parts[2].split(',')


def get_basic_layer(info, channels=None, conv_bias=False, num_segments=4):
    id = info['id']
    out_var, op, in_var = parse_expr(info['expr'])
    attr = info['attrs'] if 'attrs' in info else dict()
    if 'kernel_d' in attr.keys():
        if isinstance(attr["kernel_d"], str):
            div_num = int(attr["kernel_d"].split("/")[-1])
            attr['kernel_d'] = int(num_segments / div_num)

    out, op, in_vars = parse_expr(info['expr'])
    assert(len(out) == 1)
    assert(len(in_vars) == 1)
    if op == 'ReLU' or op == 'Pooling3d':
        mod=None
        if channels:
            out_channel = channels
        else:
            out_channel = attr['num_output']

    

    else:
        mod, out_channel, = LAYER_BUILDER_DICT[op](attr, channels, conv_bias)
    


    return id, out[0], mod, out_channel, in_vars[0]


def build_conv(attr, channels=None, conv_bias=False):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_h'], attr['kernel_w'])
    if 'pad' in attr or 'pad_w' in attr and 'pad_h' in attr:
        padding = attr['pad'] if 'pad' in attr else (attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if 'stride' in attr or 'stride_w' in attr and 'stride_h' in attr:
        stride = attr['stride'] if 'stride' in attr else (attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    conv = fluid.dygraph.Conv2D(channels, out_channels, ks, stride, padding, bias_attr=conv_bias)

    return conv, out_channels


def build_pooling(attr, channels=None, conv_bias=False):
    method = attr['mode']
    pad = attr['pad'] if 'pad' in attr else 0
    method = 'max' if method == 'max' else 'avg'
    pool = fluid.dygraph.Pool2D(attr['kernel_size'], pool_type=method,pool_stride= attr['stride'], pool_padding= pad,ceil_mode=True) # all Caffe pooling use ceil model
                            

    return pool, channels


# def build_relu(attr, channels=None, conv_bias=False):
#     return nn.ReLU(inplace=True), channels


def build_bn(attr, channels=None, conv_bias=False):
    return fluid.dygraph.BatchNorm(channels, momentum=0.1), channels


def build_linear(attr, channels=None, conv_bias=False):
    return fluid.dygraph.Linear(channels, attr['num_output']), channels


def build_dropout(attr, channels=None, conv_bias=False):
    return fluid.dygraph.Dropout(p=attr['dropout_ratio']), channels

def build_conv3d(attr, channels=None, conv_bias=False):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
    if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
        padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
        stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    conv = fluid.dygraph.Conv3D(channels, out_channels, ks, stride, padding, bias_attr=conv_bias)

    return conv, out_channels

# def build_pooling3d(attr, channels=None, conv_bias=False):
#     method = attr['mode']
#     ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
#     if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
#         padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
#     else:
#         padding = 0
#     if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
#         stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
#     else:
#         stride = 1
#     if method == 'max':
#         pool = nn.MaxPool3d(ks, stride, padding,
#                             ceil_mode=True) # all Caffe pooling use ceil model
#     elif method == 'ave':
#         pool = nn.AvgPool3d(ks, stride, padding,
#                             ceil_mode=True)  # all Caffe pooling use ceil model
#     else:
#         raise ValueError("Unknown pooling method: {}".format(method))

#     return pool, channels

def build_bn3d(attr, channels=None, conv_bias=False):
    return fluid.dygraph.BatchNorm(channels, momentum=0.1), channels


LAYER_BUILDER_DICT['Convolution'] = build_conv

LAYER_BUILDER_DICT['Pooling'] = build_pooling

# LAYER_BUILDER_DICT['ReLU'] = build_relu

LAYER_BUILDER_DICT['Dropout'] = build_dropout

LAYER_BUILDER_DICT['BN'] = build_bn

LAYER_BUILDER_DICT['InnerProduct'] = build_linear

LAYER_BUILDER_DICT['Conv3d'] = build_conv3d

# LAYER_BUILDER_DICT['Pooling3d'] = build_pooling3d

LAYER_BUILDER_DICT['BN3d'] = build_bn3d

