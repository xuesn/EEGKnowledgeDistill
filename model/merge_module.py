'''
作者：大鲸鱼crush
链接：https://juejin.cn/post/7502610991739666458
来源：稀土掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class merge_cat(nn.Module):
    def __init__(self, common_dim):
        super().__init__()
        # 为了平衡不同来源的特征，可以加个learnable weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.common_dim = common_dim
        # 再加个BatchNorm，效果会更好
        self.bn = nn.BatchNorm1d(common_dim*2)

    # features = [f1,f2]  fx: bs*feat_dim
    def forward(self, feat1, feat2 ):
        
        features = [self.alpha * feat1, self.beta * feat2]
        concat_feat = torch.cat(features, dim=1)
        
        return self.bn(concat_feat)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class merge_sum(nn.Module):
    def __init__(self, common_dim):
        super().__init__()
        # 为了平衡不同来源的特征，可以加个learnable weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.common_dim = common_dim
        # 再加个BatchNorm，效果会更好
        self.bn = nn.BatchNorm1d(common_dim)

    # features = [f1,f2]  fx: bs*feat_dim
    def forward(self, feat1, feat2 ):        

        # 加权相加，让模型自己学习最佳权重
        fused = self.alpha * feat1 + self.beta * feat2        
            
        return self.bn(fused)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class merge_mul(nn.Module):
    def __init__(self, common_dim):
        super().__init__()
        # 为了平衡不同来源的特征，可以加个learnable weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.beta = nn.Parameter(torch.tensor(0.5))

        self.common_dim = common_dim
        # 再加个BatchNorm，效果会更好
        self.bn = nn.BatchNorm1d(common_dim)
        
    # features = [f1,f2]  fx: bs*feat_dim
    def forward(self, feat1, feat2 ):        

        # 加权相加，让模型自己学习最佳权重
        fused = self.alpha * feat1 *  feat2        
        
        # 残差连接，防止信息丢失
        fused = fused + feat1
        
        return self.bn(fused)





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        # Query, Key, Value的映射
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        # 最后的输出调整
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        # 可学习的缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, feat_query, feat_kv):
        """
        feat_query: 查询特征 [B, C, H1, W1]
        feat_kv: 键值特征 [B, C, H2, W2]
        """
        B, C, H, W = feat_query.size()
        _, _, H_kv, W_kv = feat_kv.size()
        # 生成Q, K, V
        query = self.query_conv(feat_query).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        key = self.key_conv(feat_kv).view(B, -1, H_kv * W_kv)  # [B, C', H_kv*W_kv]
        value = self.value_conv(feat_kv).view(B, -1, H_kv * W_kv)  # [B, C, H_kv*W_kv]
        # 计算attention map
        attention = torch.bmm(query, key)  # [B, HW, H_kv*W_kv]
        attention = self.softmax(attention)
        # 加权求和
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        # 输出调整
        out = self.out_conv(out)
        # 残差连接
        out = self.gamma * out + feat_query
        return out

class merge_attn(nn.Module):
    def __init__(self, common_dim):
        super().__init__()

        self.common_dim = common_dim
        # 再加个BatchNorm，效果会更好
        self.bn = nn.BatchNorm1d(common_dim)

        self.CrossAttentionFusion = CrossAttentionFusion( in_channels=1, reduction=1)

    # features = [f1,f2]  fx: bs*feat_dim
    def forward(self, feat1, feat2 ):        
        feat_query = feat1.unsqueeze(dim=1).unsqueeze(dim=1)   
        feat_kv = feat2.unsqueeze(dim=1).unsqueeze(dim=1)
        """
        feat_query: 查询特征 [B, C, H1, W1]
        feat_kv: 键值特征 [B, C, H2, W2]
        """
        # HW是对于图像，由于CrossAttentionFusion内会合并HW。这里不用区分HW，只需要unsqueeze(dim=1)2次，一次C，一次H即可

        fused = self.CrossAttentionFusion(feat_query, feat_kv)
        fused = fused.reshape(fused.shape[0], -1)
        
        return self.bn(fused)


