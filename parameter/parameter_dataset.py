import logging
from torch import nn
import numpy as np



def parameter_dataset():
    # 钳位、归一化
    clamp=500
    norm_type='norm_per_sample'
    
    mask_num=1
    mask_len=1
    logging.info('mask_num:{},mask_len:{}'.format(mask_num, mask_len))

    # 数据
    sub_list = None

    return clamp, norm_type, mask_num, mask_len, sub_list

