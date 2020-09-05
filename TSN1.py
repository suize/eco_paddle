import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

import paddle.fluid.dygraph as dygraph

def convpool(in_channel,out_channel,padding=None,pooling=2,kernel = 3,act='relu'):
    if padding == None:
        padding = int((kernel-1)/2)
    layers = [
        dygraph.Conv2D(in_channel,out_channel,kernel,padding=padding,act=act),
        dygraph.BatchNorm(out_channel)
    ]
    if pooling>1:
        layers.append(dygraph.Pool2D(pooling,pool_stride=pooling))

    return fluid.dygraph.Sequential(*layers)

def convbn(in_channel,out_channel,padding=None,stride = 1,kernel = 3,act = None):
    if padding == None:
            padding = int((kernel-1)/2)
    return fluid.dygraph.Sequential(
        dygraph.Conv2D(in_channel,out_channel,kernel,stride=stride,padding=padding,act=act),
        dygraph.BatchNorm(out_channel)
    )

class TSNResNet(fluid.dygraph.Layer):
    #定义网络结构，代码补齐
    def __init__(self,name=None,num=None):
        super(TSNResNet,self).__init__()
        self.convbn = convbn(3,16)
        self.convpools = dygraph.Sequential(
            convpool(16,32,pooling = 4),
            convpool(32,64,pooling = 4),
            convpool(64,128)
            )
        self.fcs = dygraph.Sequential(
            dygraph.Linear(7*7*128,1024,act='relu'),
            dygraph.BatchNorm(1024),
            dygraph.Dropout(0.5),
            dygraph.Linear(1024,101,act='softmax')
        )
        self.seg_num = 32
    def forward(self,inputs,label=None):
        x = fluid.layers.reshape(inputs,[-1,inputs.shape[2],inputs.shape[3],inputs.shape[4]])
        x = self.convbn(x)
        x = self.convpools(x)
        x = fluid.layers.reshape(x,[x.shape[0],-1])
        
        x = fluid.layers.reshape(x,shape=[-1,self.seg_num,x.shape[1]])
        x = fluid.layers.reduce_mean(x,dim=1)
        
        x = self.fcs(x)


        # print('1'*50)
        # print('xxx:',x.shape)

        if label is not None:
            acc = fluid.layers.accuracy(input = x,label = label)
            return x,acc
        return x



        



if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = TSNResNet('resnet', 50)
        img = np.zeros([2, 32, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        label = np.zeros([2,1,101])
        outs = network(img,label).numpy()
        print(outs.shape)