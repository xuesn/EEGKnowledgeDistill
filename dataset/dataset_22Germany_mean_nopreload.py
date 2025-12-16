
from .dataset_nopreload import *
import os
import numpy as np
import json
from utils.utils import main_dir

dataset_str = '22Germany_mean'
ori_class_num = 1854  # train的标签也是从0~1853但是有200个标签没有
ori_electrode_num = 63

# 0.0s ~ 1.0s  250Hz
ori_timepoint_num=250

preproc_dir_path=main_dir + '/data/eeg_preprocess/Vision/22_Germany-merge-average-sample//'
dataset_dir = main_dir + '/data/eeg_preprocess/Vision/22_Germany-merge-average//'

def load_json_22Germany_mean(train_or_test,
        dataset_dir, sub_list, ):
   
    # 先读入第一个，timepoint和electrode数就不用输入了
    sub_name = sub_list[0]

    eeg_path_json_fname =  '_'.join([sub_name,train_or_test,'eeg_path.json'])  
    eeg_path_json_path = os.path.join(dataset_dir,sub_name,train_or_test,eeg_path_json_fname)  # 0829
    with open(eeg_path_json_path, "r",encoding='UTF-8') as f:
        eeg_path_dataset = json.load(f)

    label_0_1853_json_fname =  '_'.join([sub_name,train_or_test,'label_0_1853.json'])  
    label_0_1853_json_path = os.path.join(dataset_dir,sub_name,train_or_test,label_0_1853_json_fname)
    with open(label_0_1853_json_path, "r",encoding='UTF-8') as f:
        label_0_1853_dataset = json.load(f)

    imgNO_json_fname =  '_'.join([sub_name,train_or_test,'imgNO.json'])  
    imgNO_json_path = os.path.join(dataset_dir,sub_name,train_or_test,imgNO_json_fname)
    with open(imgNO_json_path, "r",encoding='UTF-8') as f:
        imgNO_dataset = json.load(f)

    print(sub_name,' added!!')
    
    # 再读入后续的
    if len(sub_list)>1:
        for sub_name in sub_list[1:]:

            eeg_path_json_fname =  '_'.join([sub_name,train_or_test,'eeg_path.json'])  
            eeg_path_json_path = os.path.join(dataset_dir,sub_name,train_or_test,eeg_path_json_fname)
            with open(eeg_path_json_path, "r",encoding='UTF-8') as f:
                eeg_path_dataset_this_sub = json.load(f)

            label_0_1853_json_fname =  '_'.join([sub_name,train_or_test,'label_0_1853.json'])  
            label_0_1853_json_path = os.path.join(dataset_dir,sub_name,train_or_test,label_0_1853_json_fname)
            with open(label_0_1853_json_path, "r",encoding='UTF-8') as f:
                label_0_1853_dataset_this_sub = json.load(f)

            imgNO_json_fname =  '_'.join([sub_name,train_or_test,'imgNO.json'])  
            imgNO_json_path = os.path.join(dataset_dir,sub_name,train_or_test,imgNO_json_fname)
            with open(imgNO_json_path, "r",encoding='UTF-8') as f:
                imgNO_dataset_this_sub = json.load(f)

            eeg_path_dataset += eeg_path_dataset_this_sub
            label_0_1853_dataset += label_0_1853_dataset_this_sub
            imgNO_dataset += imgNO_dataset_this_sub
            print(sub_name,' added!!')
    
    return eeg_path_dataset,label_0_1853_dataset, imgNO_dataset


def dataset_22Germany_mean(train_or_test ,
        sub_list,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):

    (eeg_path_dataset,label_0_1853_dataset, 
        imgNO_dataset) = load_json_22Germany_mean(train_or_test,
                            dataset_dir, sub_list, )        

    label_0_1853_dataset=(np.array(label_0_1853_dataset)+1).astype(np.int32)       
    imgNO_dataset=np.array(imgNO_dataset).astype(np.int32).reshape(-1,1)
    
    eeg_npy_dir = None
    label_npy_dir = None
    imgNO_npy_dir = None
    epoch_point_st = 0  # 0:0ms
    epoch_point_end = 125  # 125:500ms
    subject_list = None
    session_list = None

    whole_set = My_Dataset_nopreload(
            eeg_path_dataset,label_0_1853_dataset,imgNO_dataset,
            eeg_npy_dir, label_npy_dir,imgNO_npy_dir,
            ori_timepoint_num,ori_electrode_num,ori_class_num,
            preproc_dir_path,subject_list,session_list,
            clamp_thres,epoch_point_st,epoch_point_end,
            norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,)
        
    return whole_set


def dataset_22Germany_mean_train(
        sub_list,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):
    train_or_test = 'train'
    return dataset_22Germany_mean(train_or_test,
        sub_list, clamp_thres, norm_per_sample, norm_per_electrode, norm_per_2sample_electrode,)



