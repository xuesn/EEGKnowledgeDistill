"https://github.com/dongyangli-del/EEG_Image_decode"


import torch
from torch import nn
import math
from einops.layers.torch import Rearrange, Reduce
from torch.nn import functional as F


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


class ConvBlock(nn.Module):
    def __init__(self, num_channels, num_features):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)
        self.residual_conv = nn.Conv1d(num_channels, num_features, kernel_size=1)

    def forward(self, x):
        # print(f'ConvBlock input shape: {x.shape}')
        residual = self.residual_conv(x)
        # residual = x
        # print(f'residual shape: {residual.shape}')
        
        x = F.gelu(self.conv1(x))
        x = self.norm1(x)
        # print(f'After first convolution shape: {x.shape}')
                
        x = F.gelu(self.conv2(x))
        x = self.norm2(x)
        # print(f'After second convolution shape: {x.shape}')
        
        x = F.gelu(self.conv3(x))
        x = self.norm3(x)
        # print(f'After third convolution shape: {x.shape}')
        
        x += residual
        # print(f'ConvBlock output shape: {x.shape}')
        return x
'''
class MLPHead(nn.Module):
    def __init__(self, in_features, num_latents, dropout_rate=0.25):
        super(MLPHead, self).__init__()

        self.layer1 = nn.Sequential(
            Rearrange('B C L->B L C'),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_latents),
            nn.GELU(),
            nn.Dropout(dropout_rate), 
            Rearrange('B L C->B (C L)'),
        )
    def forward(self, x):
        # print(f'MLPHead input shape: {x.shape}')
        x = self.layer1(x)
        # print(f'After first layer of MLPHead shape: {x.shape}')
        return x
'''


class MetaEEG(nn.Module):
    def __init__(self, num_channels, sequence_length, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(MetaEEG, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)               
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.conv_blocks = nn.Sequential(*[ConvBlock(num_channels, sequence_length) for _ in range(num_blocks)],
        # 怀疑这里写错了，不应该是sequence_length，改成num_features会报错，还是用ATM作者的吧
        # self.conv_blocks = nn.Sequential(*[ConvBlock(num_channels, num_features) for _ in range(num_blocks)],
                                         Rearrange('B C L->B L C'))
        self.linear_projection = nn.Sequential(
                                            Rearrange('B L C->B C L'),
                                            nn.Linear(sequence_length, num_latents),
                                            Rearrange('B C L->B L C'))
        self.temporal_aggregation = nn.Linear(sequence_length, 1)
        # self.clip_head = MLPHead(num_latents, num_latents)
        # self.mse_head = MLPHead(num_latents, num_latents)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.01))
        # self.loss_func = ClipLoss()
        
    def forward(self, x, subject_id=0):
        # input: bs*time*ch
        x = x.transpose(2,1)


        # print(f'Input shape: {x.shape}')
        # attn_output, _ = self.attention(x, x, x)
       
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        x = self.subject_wise_linear[subject_id](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        
        x = self.conv_blocks(x)
        # print(f'After convolutional blocks shape: {x.shape}')
        
        # x = self.conv_blocks(x)
        # print(f'After convolutional blocks shape: {x.shape}')
        
        x = self.linear_projection(x)
        # print(f'After linear projection shape: {x.shape}')
        
        x = self.temporal_aggregation(x)
        # print(f'After temporal aggregation shape: {x.shape}')

        # clip_out = self.clip_head(x)
        # # print(f'Clip head output shape: {clip_out.shape}')
    
        # mse_out = self.mse_head(x)
        # # print(f'MSE head output shape: {mse_out.shape}')

        # return clip_out, mse_out
        return x
