import logging
from torch import nn
import numpy as np

import os

from utils.utils import main_dir

def parameter_save(learn_rate,train_batch_size,world_size,seed,mask_num,mask_len,dlt_ae,loss_ae_weight,temp_eeg,loss_spc_weight,margin_eeg,loss_triplet_weight,
delta_vis, temperature_vis, loss_vis_huber_weight, loss_vis_infonce_weight, loss_vis_kl_weight, vis_emb_str, vis_emb_type, vis_reweight_type, vis_layer_str,
flag_layerwise_or_reweight,
dataset_str, single_multi_cross

): 
    # ~ 运行前需修改的 ~ ~ ~ ~ ~
    # 存储路径 代码 数据集 模型
    # 98 240
    # save_share_path = '/share/models/snxue/'  # share 被cp命令卡死了 且为D 状态，无法中断
    save_share_path = main_dir+'/models/'
    save_code_dir='12-08-CMKD重跑'
    # save_dataset_dir='15Stanford_6class-singleSub'
    save_dataset_dir=dataset_str + '-' + single_multi_cross
    save_model_dir='newSCT'

    # ~ 不需设定的 ~ ~ ~ ~ ~
    # 参数信息
    # 仅限train相关，模型和数据集参数要手动设定save_model_dir和save_dataset_dir
    parameter_str_short =  'loss_' + \
                '_lr-'+str(learn_rate) + \
                '_bs-'+str(train_batch_size) + \
                '_ws-'+str(world_size) + \
                '_sd-'+str(seed)  +  \
                '_mN-'+str(mask_num)  + \
                '_mL-'+str(mask_len)  + \
                '_dlt-'+str(dlt_ae)  + \
                '_mW-'+str(loss_ae_weight)  + \
                '_tmp-'+str(temp_eeg)  + \
                '_spcW-'+str(loss_spc_weight)  + \
                '_mg-'+str(margin_eeg)  + \
                '_triW-'+str(loss_triplet_weight)  + \
                '_dVs-'+str(delta_vis) + \
                '_tVs-'+str(temperature_vis) + \
                '_vhW-'+str(loss_vis_huber_weight) + \
                '_viW-'+str(loss_vis_infonce_weight) + \
                '_vkW-'+str(loss_vis_kl_weight) + \
                '_vEm-'+vis_emb_str + \
                '_veT-'+vis_emb_type+'_vrT-'+vis_reweight_type+ \
                '-'+vis_layer_str+ \
                '-'+flag_layerwise_or_reweight

    # 文件夹路径
    save_dir_ckpt = os.path.join(save_share_path,'checkpoint',save_code_dir, save_dataset_dir, save_model_dir, parameter_str_short)
    save_dir_csv = os.path.join(save_share_path,'loss_acc',save_code_dir, save_dataset_dir, save_model_dir, parameter_str_short)
    save_dir_csv_batch = os.path.join(save_share_path,'loss_acc_batch',save_code_dir, save_dataset_dir, save_model_dir, parameter_str_short)
    return save_code_dir, save_model_dir, save_dataset_dir, parameter_str_short, save_dir_ckpt, save_dir_csv, save_dir_csv_batch




