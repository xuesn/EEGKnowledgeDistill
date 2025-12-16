"https://github.com/dongyangli-del/EEG_Image_decode"


import torch
from torch import nn


from torch import Tensor

from einops.layers.torch import Rearrange, Reduce


# 修改NICE的参数以适配不同电极数不同采样点数的数据集
class PatchEmbedding(nn.Module):
    def __init__(self, electrode_num, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 5), (1, 1)),
            # nn.AvgPool2d((1, 5), (1, 5)),  
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (electrode_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # nn.Dropout(0.5),
        )#nn.AvgPool2d((1, 51), (1, 5)), 改小

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


# class Enc_eeg(nn.Sequential):
#     def __init__(self, emb_size=40, **kwargs):
#         super().__init__(
#             PatchEmbedding(emb_size),
#             FlattenHead()
#         )
class Enc_eeg(nn.Sequential):
    def __init__(self, electrode_num, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(electrode_num, emb_size),
            FlattenHead()
        )

'''
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
'''

'''
class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 
'''

class NICE(nn.Module):
    def __init__(self,electrode_num,a_dim):
        super().__init__()
        # self.enc_eeg = Enc_eeg()
        # self.proj_eeg = Proj_eeg()
        self.enc_eeg = Enc_eeg(electrode_num,emb_size=a_dim,)
        # self.proj_eeg = Proj_eeg(embedding_dim=a_dim)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.loss_func = ClipLoss()        
    def forward(self, data):
        # input: bs*time*ch
        data = data.transpose(2,1)
        eeg_embedding = self.enc_eeg(data)
        # out = self.proj_eeg(eeg_embedding)

        # return out  
        return eeg_embedding  
      