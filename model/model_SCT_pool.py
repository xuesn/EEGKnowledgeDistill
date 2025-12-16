# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
import math
import torch



class model_SCT_pool(nn.Module):
    # kernal1-3是时间维度1维卷积；kernal4是所有电极加权
    def __init__(self, ch1=16, ch2=32, ch3=64, ch4=64,
                 kernal1=5, kernal2=5, kernal3=5, kernal4=124,
                 pool_stride=2,
                 dropout_conv=0.25,
                 classify_fc_hid=16, class_num=6,
                 activate_type='elu',
                 pool_type='mean',
                 timepoint_num=32,electrode_num=124,                   
                 trans_layers=1, trans_fc_hid=64,  trans_head=4,
                 act_func_trans='relu', dropout_trans=0.1,
                 ch_dim_reduce=16,
                 decoder_depth=0, decoder_embed_dim=64, decoder_num_heads=1,
                 vis_emb_dim=512 ):  # 换新数据集了 class_num=6类或72张
        super(model_SCT_pool, self).__init__()

        self.timepoint_num = timepoint_num
        self.electrode_num = electrode_num

    # 选择激活函数--------------------------------------------------
        if activate_type == 'relu':
            self.activate_func = nn.ReLU()
        elif activate_type == 'leaky-relu':
            self.activate_func = nn.LeakyReLU(negative_slope=0.1)
        elif activate_type == 'elu':
            self.activate_func = nn.ELU(alpha=1.0)
        # elif activate_type == 'glu':
        #     self.activate_func = nn.GLU()#RuntimeError: Halving dimension must be even, but dimension 3 is size 31
        elif activate_type == 'gelu':
            self.activate_func = nn.GELU()
        else:
            print('!!Undefined Activate Function!!')
            self.activate_func = nn.ELU()

    # 选择pooling方式--------------------------------------------------
        if pool_type == 'max':
            self.pool_layer = nn.MaxPool2d(kernel_size=(pool_stride, 1),
                                           stride=(pool_stride, 1))
        elif pool_type == 'mean':
            self.pool_layer = nn.AvgPool2d(kernel_size=(pool_stride, 1),
                                           stride=(pool_stride, 1))

    # 第一层卷积--对所有电极重新加权组合，forward中注意调整数据维度顺序--------------------------------------------------
        self.reweight_layer = nn.Sequential(
            # 简单整合所有电极信息，forward中再调整数据维度顺序
            nn.Conv2d(in_channels=1, out_channels=electrode_num,
                      kernel_size=(1, electrode_num), stride=(1, 1)),
        )

    # 用卷积编码特征--对时间维度卷积，最后整合所有电极信息，再输入Transformer编码器--------------------------------------------------
        self.conv_encoder_time = nn.Sequential(
            # layer1
            nn.Conv2d(in_channels=1, out_channels=ch1,
                      kernel_size=(kernal1, 1), stride=(1, 1),),
                    #   padding='same', padding_mode='zeros'),  # padding用str控制时doesn't support any stride values other than 1.
            nn.BatchNorm2d(num_features=ch1),
            self.activate_func,
            # layer2
            # nn.Dropout(p=dropout_conv),
            nn.Conv2d(in_channels=ch1, out_channels=ch2,
                      kernel_size=(kernal2, 1), stride=(1, 1),),
                    #   padding='same', padding_mode='zeros'),
            nn.BatchNorm2d(num_features=ch2),
            self.activate_func,
            self.pool_layer,
            # layer3
            nn.Dropout(p=dropout_conv),
            nn.Conv2d(in_channels=ch2, out_channels=ch3,
                      kernel_size=(kernal3, 1), stride=(1, 1),),
                    #   padding='same', padding_mode='zeros'),
            nn.BatchNorm2d(num_features=ch3),
            self.activate_func,
            self.pool_layer,
        )
        # 加跨度为1层skip，以后可以试试跨度为2层、3层的
        self.conv_encoder_skip1 = nn.Sequential(
            nn.Conv2d(in_channels=ch3, out_channels=2*ch3,
                      kernel_size=(3, 1), stride=(1, 1),
                      padding='same', padding_mode='zeros'),
            nn.BatchNorm2d(num_features=2*ch3),
            self.activate_func,
            nn.Conv2d(in_channels=2*ch3, out_channels=ch3,
                      kernel_size=(3, 1), stride=(1, 1),
                      padding='same', padding_mode='zeros'),
            nn.BatchNorm2d(num_features=ch3),
            self.activate_func,
        )
        # self.conv_encoder_skip2 = nn.Sequential(
        #     nn.Conv2d(in_channels=ch3, out_channels=2*ch3,
        #               kernel_size=(3, 1), stride=(1, 1),
        #               padding='same', padding_mode='zeros'),
        #     nn.BatchNorm2d(num_features=2*ch3),
        #     self.activate_func,
        #     nn.Conv2d(in_channels=2*ch3, out_channels=ch3,
        #               kernel_size=(3, 1), stride=(1, 1),
        #               padding='same', padding_mode='zeros'),
        #     nn.BatchNorm2d(num_features=ch3),
        #     self.activate_func,
        # )
        # 整合所有电极的信息
        kernal4 = electrode_num
        self.conv_encoder_electrode = nn.Sequential(
            nn.Dropout(p=dropout_conv),
            nn.Conv2d(in_channels=ch3, out_channels=ch4,
                      kernel_size=(1, kernal4), stride=(1, 1)),
            nn.BatchNorm2d(num_features=ch4),
            self.activate_func,
        )
        # output.shape
        # # torch.Size([5, 64, 4, 1]) 

    # conv局部时序特征提取后，再接Transformer做全局时序特征提取--------------------------------------------------
        # 不调的参数
        layernorm_eps_trans = 1e-05
        norm_first_bool = False
        # conv_out_dim要为2的幂次
        conv_out_dim = ch4
        trans_enc_layer = nn.TransformerEncoderLayer(d_model=conv_out_dim, nhead=trans_head, dim_feedforward=trans_fc_hid,
                                                     dropout=dropout_trans, activation=act_func_trans, layer_norm_eps=layernorm_eps_trans,
                                                     batch_first=True, norm_first=norm_first_bool, )
        self.trans_tile = nn.TransformerEncoder(trans_enc_layer, num_layers=trans_layers)



    # 降维（时间维度加权），再接fc分类--------------------------------------------------
        '''conv_out_time_dim = 4 #32 timepoint
        conv_out_time_dim = 27 #125 timepoint
        conv_out_time_dim = 113 #125 timepoint  不pooling
        conv_out_time_dim = 127 #21purdue 525 timepoint '''
        def compute_conv_out_time_dim_pooling(kernal1, kernal2, kernal3, timepoint_num):
            conv_out_time_dim_pooling = timepoint_num-kernal1+1
            # layer2 3 加了pooling
            import math
            conv_out_time_dim_pooling = math.floor(conv_out_time_dim_pooling-kernal2+1)/2
            conv_out_time_dim_pooling = math.floor(conv_out_time_dim_pooling-kernal3+1)/2            
            return int(conv_out_time_dim_pooling)
        conv_out_time_dim = compute_conv_out_time_dim_pooling(kernal1, kernal2, kernal3, timepoint_num)
        self.dim_reduce_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=ch_dim_reduce,
                        kernel_size=(conv_out_time_dim, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=ch_dim_reduce),
            self.activate_func,
        )

   
    # decoder恢复原始信号--------------------------------------------------
        trans_dec_layer = nn.TransformerEncoderLayer(d_model=conv_out_dim, nhead=decoder_num_heads, dim_feedforward=decoder_embed_dim,
                                                        dropout=dropout_trans, activation=act_func_trans, layer_norm_eps=layernorm_eps_trans,
                                                        batch_first=True, norm_first=norm_first_bool, )
        '''
        # This error indicates that your module has parameters that were not used in producing loss.
        # self.dec_trans_tile = nn.TransformerEncoder(trans_dec_layer, num_layers=decoder_depth)
        '''
        self.decoder_depth = decoder_depth
        self.dec_norm = nn.LayerNorm(conv_out_dim)
        # self.dec_linear = nn.Linear(conv_out_dim, electrode_num) 
        self.dec_linear = nn.Linear(ch_dim_reduce*conv_out_dim, timepoint_num*electrode_num) 
        # self.dec_linear = nn.Sequential(
        #             nn.Linear(ch_dim_reduce*conv_out_dim, 1024),
        #             nn.BatchNorm1d(num_features=1024),
        #             nn.Dropout(p=dropout_fc),
        #             self.activate_func,
        #             nn.Linear(1024, timepoint_num*electrode_num),
        #     )

    # linear映射特征维度为vis_emb_dim--------------------------------------------------
        # self.vis_linear = nn.Linear(ch_dim_reduce*conv_out_dim, vis_emb_dim) 


    def forward(self, in_data,):
        # subNO：根据不同被试，选择不同的subject-layer。*这样要求一个batch内必须是同一被试的数据，因此训练时决定每个被试轮流训一个epoch。
        # in_data: batch,32, 124
        in_data_float = in_data.to(torch.float32)
        in_data_unsqueeze = in_data_float.unsqueeze(dim=1)
        # batch 1 32 124

        # subject-layer-adapter
        reweighted_data = self.reweight_layer(in_data_unsqueeze)
        # batch ch time electrode
        # batch 124 32 1
        # torch.transpose (input, dim0, dim1) 函数将输入张量 input 的第 dim0 个维度和第 dim1 个维度进行交换，并将交换维度后的张量返回。
        reweighted_data = torch.transpose(reweighted_data, dim0=1, dim1=3)
        # batch 1 32 124

        # conv-encode
        # conv_out = self.conv_4layers(reweighted_data)        
        # conv-encode(时间维度卷积，padding=same，不改变时间序列维度)
        conv_out1 = self.conv_encoder_time(reweighted_data)
        conv_out2 = self.conv_encoder_skip1(conv_out1)
        # conv_out3 = self.conv_encoder_skip2(conv_out1+conv_out2)
        # conv_out4 = self.conv_encoder_skip3(conv_out2+conv_out3)
        conv_out = self.conv_encoder_electrode(conv_out1+conv_out2)
        # batch  64, 4, 1

        # trans-encoder
        # batch_first: If ``True``, then the input and output tensors are provided
        #     as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        conv_out_squeeze = torch.squeeze(conv_out, dim=3)
        conv_out_permute = conv_out_squeeze.permute(0, 2, 1)
        # batch 4 ch_out
        trans_out = self.trans_tile(conv_out_permute)
        # 时间维度只剩4了，后面可以取消pooling，并用padding-conv试试

        # print(trans_out.shape)
        # torch.Size([799, 3, 64])

        # 降维
        trans_out_unsqueeze = trans_out.unsqueeze(dim=1)
        # batch 1 300 ch_ori
        dim_reduced_out = self.dim_reduce_layer(trans_out_unsqueeze)
        # batch ch_new 1 ch_ori
        # 如果15S的效果不好，可以把dim_reduced去掉
        

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
        dec_out_normed = self.dec_norm(dim_reduced_out)

        emb_reshaped = dec_out_normed.reshape(dim_reduced_out.shape[0], -1)
        eeg_recovered = self.dec_linear(emb_reshaped)
        eeg_recovered = eeg_recovered.reshape(dim_reduced_out.shape[0], self.timepoint_num,self.electrode_num)




        # 归一化可能是必要的，不然相似度矩阵值特别大，取exp后变成1和0，再取log就会出现nan
        # 应该对特征的模归一化，不然乘起来还是很大，即z-score后除上特征维度
        emb = dim_reduced_out.reshape(dim_reduced_out.shape[0], -1)
        emb_mean = torch.mean(emb, dim=1)
        emb_std = torch.std(emb, dim=1)
        emb_z_score = (emb-emb_mean.reshape(-1, 1))/(emb_std.reshape(-1, 1)+1e-5)
        feat_dim = emb.shape[1]
        emb_norm = emb_z_score/((feat_dim-1)**(1/2)+1e-5)


        # # 视觉监督 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        # linear_out = self.vis_linear(emb_reshaped)


        return  dec_out_normed, emb_norm