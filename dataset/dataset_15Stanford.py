
from .dataset_preload import *
import scipy.io

import os
import numpy as np


from utils.utils import main_dir



mat_dir = main_dir + '/eeg_datasets/15Stanford/' 

ori_timepoint_num=32# # 0.5s   62.5Hz  
ori_electrode_num=124


def load_mat_15_Stanford_single_sub_imgNO(sub_str,class_num):
    # load
    path_sub = mat_dir+sub_str+'.mat'
    data_sub = scipy.io.loadmat(path_sub)
    exemplarLabels = data_sub['exemplarLabels']  # 72张图片 1*5188（5188这个数不同被试不同）
    categoryLabels = data_sub['categoryLabels']  # 6大类 1*5188
    X_3D = data_sub['X_3D']  # 124*32*5188
    # 数据
    sample_time_electrode = X_3D.transpose(2, 1, 0)
    sample_num, timepoint_num, electrode_num = sample_time_electrode.shape
    # # 标签
    # if class_num == 6:
    #     sample_label = categoryLabels.T
    #     class_num = np.max(categoryLabels)
    #     # sample_label_onehot = np.zeros([sample_num, class_num])
    #     # sample_label_onehot[np.arange(sample_label.size), (sample_label-1).reshape([-1, ])] = 1
    # if class_num == 72:
    #     sample_label = exemplarLabels.T
    #     class_num = np.max(exemplarLabels)
    #     # sample_label_onehot = np.zeros([sample_num, class_num])
    #     # sample_label_onehot[np.arange(sample_label.size), (sample_label-1).reshape([-1, ])] = 1
    return sample_time_electrode, categoryLabels.T, exemplarLabels.T

def load_mat_15_Stanford_imgNO(subject_list,class_num):
    # 初始化第一个sub
    sub_str = subject_list[0]
    eeg_dataset, label_dataset, imgNO_dataset = load_mat_15_Stanford_single_sub_imgNO(sub_str,class_num)
    if len(subject_list)>1:
        # 逐被试append进list
        for sub_str in subject_list[1:]:
            # load
            eeg_dataset_this_sub, label_dataset_this_sub, imgNO_dataset_this_sub = load_mat_15_Stanford_single_sub_imgNO(sub_str,class_num)  
            eeg_dataset = np.concatenate([eeg_dataset, eeg_dataset_this_sub],axis=0)  
            label_dataset = np.concatenate([label_dataset, label_dataset_this_sub],axis=0)    
            imgNO_dataset = np.concatenate([imgNO_dataset, imgNO_dataset_this_sub],axis=0)    
    return eeg_dataset,label_dataset,imgNO_dataset


        # sub_list_list = [['S1'],['S2'],['S3'],['S4'],['S5'],
        #                  ['S6'],['S7'],['S8'],['S9'],['S10']]

def dataset_15Stanford_6class(subject_list,clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):
    class_num = 6 # 其实这里也没用
    eeg_dataset,label_dataset,imgNO_dataset = load_mat_15_Stanford_imgNO(subject_list,class_num)

# (5188, 32, 124)
# (5188, 1)
# np.max(label_dataset) 6
# (5188, 1)


    # label_dataset需要是1~6
    # imgNO_dataset需要是0~71
    imgNO_dataset = imgNO_dataset-1
    whole_set = My_Dataset_array_preload(
                eeg_dataset,label_dataset,imgNO_dataset,
                mat_dir, mat_dir,mat_dir,
                ori_timepoint_num,ori_electrode_num,                class_num,
                None, subject_list,None,
                clamp_thres,None,None,            
                norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,)
                
        
    return whole_set



def dataset_15Stanford_72pic(subject_list,clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,
        ):
    class_num = 72 # 其实这里也没用
    eeg_dataset,label_dataset,imgNO_dataset = load_mat_15_Stanford_imgNO(subject_list,class_num)

# (5188, 32, 124)
# (5188, 1)
# np.max(label_dataset) 6
# (5188, 1)


    # label_dataset需要是1~72
    label_dataset = np.copy(imgNO_dataset)
    # imgNO_dataset需要是0~71
    imgNO_dataset = imgNO_dataset-1
    whole_set = My_Dataset_array_preload(
                eeg_dataset,label_dataset,
                imgNO_dataset,
                mat_dir, mat_dir,
                mat_dir,
                ori_timepoint_num,ori_electrode_num,class_num,
                None, subject_list,None,
                clamp_thres,None,None,            
                norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,
               )
                
        
    return whole_set
