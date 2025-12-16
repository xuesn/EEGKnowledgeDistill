# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
import math
import torch



class model_mae(nn.Module):
    # kernal1-3是时间维度1维卷积；kernal4是所有电极加权
    def __init__(self, 
                 timepoint_num=32,electrode_num=124,                   
                 trans_layers=4, trans_fc_hid=256,  trans_head=4,
                 act_func_trans='relu', dropout_trans=0.1,
                 decoder_depth=2, decoder_embed_dim=256, decoder_num_heads=4, ):  
        super(model_mae, self).__init__()

        self.timepoint_num = timepoint_num
        self.electrode_num = electrode_num


    # encoder--------------------------------------------------
        # 不调的参数
        layernorm_eps_trans = 1e-05
        norm_first_bool = False
        # closest_ch_num要为2的幂次
        trans_enc_layer = nn.TransformerEncoderLayer(d_model=electrode_num, nhead=trans_head, dim_feedforward=trans_fc_hid,
                        dropout=dropout_trans, activation=act_func_trans, layer_norm_eps=layernorm_eps_trans,
                        batch_first=True, norm_first=norm_first_bool, )
        self.trans_tile = nn.TransformerEncoder(trans_enc_layer, num_layers=trans_layers)

        self.enc_norm = nn.LayerNorm(electrode_num)


   
    # decoder恢复原始信号--------------------------------------------------
        trans_dec_layer = nn.TransformerEncoderLayer(d_model=electrode_num, nhead=decoder_num_heads, dim_feedforward=decoder_embed_dim,
                                                        dropout=dropout_trans, activation=act_func_trans, layer_norm_eps=layernorm_eps_trans,
                                                        batch_first=True, norm_first=norm_first_bool, )
        self.decoder_depth = decoder_depth
        self.dec_trans_tile = nn.TransformerEncoder(trans_dec_layer, num_layers=decoder_depth)

        # self.dec_norm = nn.LayerNorm(electrode_num)


    def forward(self, in_data,):
        # subNO：根据不同被试，选择不同的subject-layer。*这样要求一个batch内必须是同一被试的数据，因此训练时决定每个被试轮流训一个epoch。
        # in_data: batch,32, 124
        in_data_float = in_data.to(torch.float32)
        # batch time channel

        # trans-encoder
        # batch_first: If ``True``, then the input and output tensors are provided
        #     as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        trans_out = self.trans_tile(in_data_float)

        # print(trans_out.shape)
        # torch.Size([799, 3, 64])

        enc_out_normed = self.enc_norm(trans_out)

        # 1、mask 信号恢复 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        # Decoder
        '''
        This error indicates that your module has parameters that were not used in producing loss.
        if self.decoder_depth>0:     # # 暂不使用tran作为decoder
            dec_out = self.dec_trans_tile(dim_reduced_out)
            dec_out_normed = self.dec_norm(dec_out)
        else:
            dec_out_normed = self.dec_norm(dim_reduced_out)
        '''
        dec_out = self.dec_trans_tile(trans_out)
        # dec_out_normed = self.dec_norm(dec_out)

        # 归一化可能是必要的，不然相似度矩阵值特别大，取exp后变成1和0，再取log就会出现nan
        # 应该对特征的模归一化，不然乘起来还是很大，即z-score后除上特征维度
        emb = enc_out_normed.reshape(enc_out_normed.shape[0], -1)
        emb_mean = torch.mean(emb, dim=1)
        emb_std = torch.std(emb, dim=1)
        emb_z_score = (emb-emb_mean.reshape(-1, 1))/(emb_std.reshape(-1, 1)+1e-5)
        feat_dim = emb.shape[1]
        emb_norm = emb_z_score/((feat_dim-1)**(1/2)+1e-5)

        return  emb_norm, dec_out
