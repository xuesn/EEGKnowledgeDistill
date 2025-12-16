from .dataset_nopreload import *

import os
import numpy as np

import json

from utils.utils import main_dir

dataset_str = '21Purdue'
ori_class_num = 40
ori_electrode_num=96

eeg_path_json_path = main_dir + '/data/merge_json_Vision/21_Purdue_json/'+\
    '21_Purdue_eeg_path_dataset_dict.json'
label_json_path = main_dir + '/data/merge_json_Vision/21_Purdue_json/'+\
    '21_Purdue_label_dataset_dict.json'  
imgNO_npy_path = main_dir + '/data/eeg_preprocess/21Purdue-img/imgNO_dataset_.npy'
preproc_dir_path=main_dir + '/data/eeg_preprocess/21Purdue/'+\
    '08_10_250Hz_ep-100_2000ms_bc-100_0ms_bp0p1_100Hz_amp1000000/'
# 2.1s
# 250Hz
ori_timepoint_num=525

def load_21Purdue_path():
    
    with open(eeg_path_json_path, "r",encoding='UTF-8') as f:
        eeg_path_dataset = json.load(f)['eeg_path_dataset'] 
    with open(label_json_path, "r",encoding='UTF-8') as f:
        label_dataset = json.load(f)['label_dataset']      
    imgNO_dataset = np.load(imgNO_npy_path)

    return eeg_path_dataset,label_dataset, imgNO_dataset

def dataset_21Purdue(
        subject_list,clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):
    
    eeg_path_dataset,label_dataset, imgNO_dataset = load_21Purdue_path()        
    label_dataset=np.array(label_dataset).astype(np.int16)       
    imgNO_dataset=np.array(imgNO_dataset).astype(np.int32)
    
    eeg_npy_dir = None
    label_npy_dir = None
    imgNO_npy_dir = None
    epoch_point_st = None
    epoch_point_end = None
    subject_list = None
    session_list = None
    whole_set = My_Dataset_nopreload(
            eeg_path_dataset,label_dataset,imgNO_dataset,
            eeg_npy_dir, label_npy_dir,imgNO_npy_dir,
            ori_timepoint_num,ori_electrode_num,ori_class_num,
            preproc_dir_path,subject_list,session_list,
            clamp_thres,epoch_point_st,epoch_point_end,
            norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,)
        
    return whole_set




