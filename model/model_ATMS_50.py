"https://github.com/dongyangli-del/EEG_Image_decode"


import torch
from torch import nn

from torch import Tensor

import math
from einops.layers.torch import Rearrange, Reduce




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # print(d_model) 124 96 63
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        if d_model%2==0:     # 网上搜的PositionalEncoding  
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
        elif d_model%2==1: # ATM的PositionalEncoding
            div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.d_model%2==0:   # 网上搜的PositionalEncoding
            return x + self.pe[:x.size(0), :]
        elif self.d_model%2==1:   # ATM的PositionalEncoding
            pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
            x = x + pe
            return x



class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

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

class ATMS_50(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS_50, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        # self.enc_eeg = Enc_eeg()
        # self.proj_eeg = Proj_eeg()   
        self.enc_eeg = Enc_eeg(num_channels,emb_size=num_features,)
        # self.proj_eeg = Proj_eeg(embedding_dim=num_features)     
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.loss_func = ClipLoss()       
         
    def forward(self, x):
        # input: bs*time*ch
        x = x.transpose(2,1)
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        
        # out = self.proj_eeg(eeg_embedding)
        # return out  
        return eeg_embedding  
    









