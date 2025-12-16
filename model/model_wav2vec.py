
# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
import math
import torch
import numpy as np



# wav2vec适合语音脑电，要求信号有连续语义，而我们的任务范式则是非连续的

final_dim = 512#15S
final_dim = 2048#21P
final_dim = 2048#22G

class model_wav2vec(nn.Module):
    # kernal1-3是时间维度1维卷积；kernal4是所有电极加权
    def __init__(self, 
                 final_dim = 64,logit_temp = 0.05,


                 timepoint_num=32,electrode_num=124,                   
                 trans_layers=4, trans_fc_hid=256,  trans_head=4,
                 act_func_trans='relu', dropout_trans=0.1,
                 ):  
        super(model_wav2vec, self).__init__()

        self.print_flag = True   # PT时观察用
        # self.print_flag = False  # 0509 FT

        
        self.logit_temp = logit_temp

        self.final_dim = final_dim

        # self.mask_flag = mask_flag  # 改成由外部调用forward时，输入是否mask


        self.timepoint_num = timepoint_num
        self.electrode_num = electrode_num
    
    
        
    # Quantization module--------------------------------------------------
      
        self.project_q = nn.Linear(electrode_num, final_dim)



    # encoder--------------------------------------------------
        # 不调的参数
        layernorm_eps_trans = 1e-05
        norm_first_bool = False
        # closest_ch_num要为2的幂次
        trans_enc_layer = nn.TransformerEncoderLayer(d_model=electrode_num, nhead=trans_head, dim_feedforward=trans_fc_hid,
                        dropout=dropout_trans, activation=act_func_trans, layer_norm_eps=layernorm_eps_trans,
                        batch_first=True, norm_first=norm_first_bool, )
        self.trans_tile = nn.TransformerEncoder(trans_enc_layer, num_layers=trans_layers)

        # self.enc_norm = nn.LayerNorm(electrode_num)
        self.enc_norm = nn.LayerNorm(electrode_num)

        self.final_proj = nn.Linear(electrode_num, final_dim)

    def apply_mask_(
        self,
        x,
        mask_num):       
        mask_indices = None 
        B, T, C = x.shape
        if mask_num>0:
            #  mask_type == 'time':
            # mask_st_list = np.random.randint(low=0,high=self.timepoint_num,size=mask_num)
            mask_st_list = np.random.choice(self.timepoint_num,mask_num, replace=False)
            for mask_st in mask_st_list:
                x[:,mask_st,:] = 0  # mask所有sample的相同时刻点，每次mask的是全电极，相当于mask掉一张某个时刻点的电极图
            mask_indices = mask_st_list
        return x, mask_indices

    def forward(self, in_data,mask=False,mask_num=0):
        mask_indices = None
        in_data_float = in_data.to(torch.float32)
        bs,t_num,c_num = in_data_float.shape
        # bs*t*c
        if self.print_flag:
            print('in_data_float:',in_data_float.shape)

        if mask and mask_num>0:
            # print(bs_crop_feat.device)
            in_feat_masked,mask_indices = self.apply_mask_(in_data_float,mask_num)  
        else:
            in_feat_masked = in_data_float

        # trans-encoder
        trans_out = self.trans_tile(in_feat_masked)
        enc_out_normed = self.enc_norm(trans_out)
        
        if mask and mask_num>0:
            # VQ映射
            # 不能用所有其他timepoint作为负样本，太多了
            masked_in_ = in_data_float[:,mask_indices,:]
            unmask_indices = [i for i in range(t_num) if not i in mask_indices]

            # 选最多20个负样本 ， 不足则全选  （mask后的时刻点也互为负样本）        
            neg_sample_num = 20
            if len(unmask_indices)<20:
                neg_sample_num = len(unmask_indices)
            unmask_20timepoint_indices = np.random.choice(unmask_indices,neg_sample_num,replace = False)
            unmasked_in_ = in_data_float[:,unmask_20timepoint_indices,:]

            pos_vq_feat = self.project_q(torch.reshape(masked_in_,[-1,c_num]))
            neg_vq_feat = self.project_q(torch.reshape(unmasked_in_,[-1,c_num]))
            # (bs.* mask/unmask_num ) *final_dim
            neg_vq_feat_reshape = torch.reshape(neg_vq_feat,[bs,neg_sample_num,self.final_dim])
            # bs * neg_num * final_dim
            neg_vq_feat_repeatM = neg_vq_feat_reshape.repeat_interleave(mask_num,dim=0)
            # (bs.*mask_num) * neg_num * final_dim

            # - * - - * - - * - - * - - * - - * - - * - - * - - * - - * -
            # 只对mask的输入做linear，其他的不用于计算对比loss
            bs,crop_num,feat_dim = enc_out_normed.shape
            mask_emb = enc_out_normed[:,mask_indices,:]
            # bs * mask_num * feat_dim（electrode_num)

            mask_emb_reshape = torch.reshape(mask_emb,[bs * mask_num , feat_dim])
            # (bs.*mask_num) * feat_dim

            pos_linear_out = self.final_proj(mask_emb_reshape)
            # (bs.*mask_num) *final_dim


            # - * - - * - - * - - * - - * - - * - - * - - * - - * - - * -
            # (bs.*mask_num) *final_dim
            # (bs.*mask_num) *final_dim
            # (bs.*mask_num) * neg_num * final_dim


            loss_contrast = self.compute_preds_then_loss(pos_linear_out,  
                                            pos_vq_feat,
                                            neg_vq_feat_repeatM,)
        else:
            # linear_out_feat_mask = None
            loss_contrast = None
            

        # return vq_feat_reshape,enc_out_normed,linear_out_feat,mask_indices
        return enc_out_normed, loss_contrast

    # 令正样本始终为第一个，label直接就是(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    def compute_preds_then_loss(self, 
                pos_linear_out,  
                pos_vq_feat,
                neg_vq_feat_repeatM,):
        if self.print_flag:
            print('pos_linear_out:',pos_linear_out.shape)
            print('pos_vq_feat:',pos_vq_feat.shape)
            print('neg_vq_feat_repeatM:',neg_vq_feat_repeatM.shape)
        pos_vq_feat = pos_vq_feat.unsqueeze(1)
        if self.print_flag:
            print('pos_vq_feat:',pos_vq_feat.shape)
        targets = torch.cat([pos_vq_feat, neg_vq_feat_repeatM], dim=1)
        if self.print_flag:
            print('targets:',targets.shape)

        pos_linear_out = pos_linear_out.unsqueeze(1)
        logits = torch.cosine_similarity(pos_linear_out.float(), targets.float(), dim=-1)
        if self.print_flag:
            print('logits:',logits.shape)

        logits = logits / self.logit_temp
        logits = logits.type_as(pos_linear_out)

        if self.print_flag:
            print('logits:',logits[0:3,:])

        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        # 这里数值稳定性保持的不好，应该提前 对每个样本的特征做模归一化  即z-score后除上特征维度
        exp_dot_tempered = (
            torch.exp(logits -
                      torch.max(logits, dim=1, keepdim=True)[0]) + 1e-5
        )
        if self.print_flag:
            print('exp_dot_tempered:',exp_dot_tempered.shape)
            print('exp_dot_tempered[0:3,:]:',exp_dot_tempered[0:3,:])


        prob_ = exp_dot_tempered / (torch.sum(exp_dot_tempered , dim=1, keepdim=True))
        if self.print_flag:
            print('prob_:',prob_.shape)
        # print('prob_[0:3,:]:',prob_[0:3,:])
        print('prob_[0,:]:',prob_[0,:])

        log_prob = -torch.log(prob_)
        if self.print_flag:
            print('log_prob:',log_prob.shape)
            print('log_prob[0:3,:]:',log_prob[0:3,:])

                
        # 仅第一列为正样本
        supervised_contrastive_loss = torch.mean(log_prob[:,0])
        if self.print_flag:
            print('supervised_contrastive_loss:',supervised_contrastive_loss)



        self.print_flag = False
        return supervised_contrastive_loss








