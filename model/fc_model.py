

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 如果映射网络只包含Encoder的话，那么特征表示里会有很多预训练任务相关特征，会影响下游任务效果；
# 而加上Projector，等于增加了网络层深，这些任务相关特征就聚集在Projector，
# 此时Encoder则不再包含预训练任务相关特征，只包含更通用的细节特征。这是为何需要两次映射过程，知乎猜大致是这个原因，但也是猜测，不保证正确性。
class ProjNet_FC(nn.Module):#目前就用FC做特征映射，改变特征分布，不用其他结构
    def __init__(self,
                in_dim,out_dim,
                hid_dim_list,
                activate_func='relu',
                whether_last_layer_act=True,
                last_layer_act_func='softmax',
                dropout_fc=0):
        super(ProjNet_FC, self).__init__()
        #多少隐层
        hid_layer_num=len(hid_dim_list)
        # 选择激活函数--------------------------------------------------
        if activate_func == 'relu':
            activate_func = nn.ReLU(inplace=True)
        elif activate_func == 'leaky-relu':
            activate_func = nn.LeakyReLU(negative_slope=0.1)
        elif activate_func == 'elu':
            activate_func = nn.ELU(alpha=1.0)
        # elif activate_func == 'glu':
        #     activate_func = nn.GLU()#RuntimeError: Halving dimension must be even, but dimension 3 is size 31
        elif activate_func == 'gelu':
            activate_func = nn.GELU()
        else:
            print('!!Undefined Activate Function!!')
            activate_func = nn.ELU()
        # 选择最后一层的激活函数--------------------------------------------------
        if last_layer_act_func == 'softmax':#作为分类头时使用Softmax,作为project头时用其他激活函数或不加激活
            last_layer_act_func =nn.Softmax(dim=-1)
        elif last_layer_act_func == 'relu':
            last_layer_act_func = nn.ReLU(inplace=True)
        elif last_layer_act_func == 'leaky-relu':
            last_layer_act_func = nn.LeakyReLU(negative_slope=0.1)
        elif last_layer_act_func == 'elu':
            last_layer_act_func = nn.ELU(alpha=1.0)
        # elif last_layer_act_func == 'glu':
        #     last_layer_act_func = nn.GLU()#RuntimeError: Halving dimension must be even, but dimension 3 is size 31
        elif last_layer_act_func == 'gelu':
            last_layer_act_func = nn.GELU()
        else:
            print('!!Undefined Activate Function!!')
            last_layer_act_func = nn.ELU()

        # FC作为project模型--------------------------------------------------
        self.model = nn.Sequential()
        layer_in_dim=in_dim
        #添加最后一层以外的线性层和激活函数
        for layerNO,layer_out_dim in enumerate(hid_dim_list):#hid_dim_list为空时，enumerate会报错吗？答：不会
        # for layerNO in range(0,hid_layer_num):
            # layer_out_dim=hid_dim_list[layerNO]
            layer=nn.Sequential(
                    nn.Linear(layer_in_dim, layer_out_dim),
                    nn.BatchNorm1d(num_features=layer_out_dim, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True, device=None, dtype=None),
                    nn.Dropout(p=dropout_fc),
                    activate_func,
            )
            self.model.add_module("layer"+str(layerNO), layer)
            layer_in_dim=layer_out_dim
        #添加最后一层
        layer=nn.Sequential(
                nn.Linear(layer_in_dim, out_dim),
        )
        self.model.add_module("last_linear", layer)
        #最后一层是否添加激活函数
        if whether_last_layer_act:
            layer=nn.Sequential(
                # nn.BatchNorm1d(num_features=out_dim),#如果是softmax，可能不应加batchnorm，这里统一都不加了
                last_layer_act_func,
            )
            self.model.add_module('last_layer_act', layer)


    # @autocast()
    def forward(self, cls_token):
        
        # cls_token=emb_vit[:,0,:]
        # 拉成1维，接fc
        fc_in = cls_token.reshape(cls_token.shape[0], -1)
        # print('fc输入特征的维度大小:', fc_in.shape[1])
        fc_out = self.model(fc_in)
        # prob = F.softmax(fc_out, dim=1)  #24-04-12:prob多过了一层softmax，改回来再试试 
        prob = fc_out.squeeze(1)  # 移除数组中维度为1的维度
        return prob







