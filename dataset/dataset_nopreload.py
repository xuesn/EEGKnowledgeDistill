import os
import random
import numpy as np
import torch
import logging

from .utils_dataset import *


class My_Dataset_nopreload(torch.utils.data.Dataset):

    #主构造函数
    def __init__(self,
            eeg_path_dataset=None,label_dataset=None,imgNO_dataset=None,
            eeg_npy_dir=None, label_npy_dir=None,imgNO_npy_dir=None,
            ori_timepoint_num=None,ori_electrode_num=None,ori_class_num=None,
            preproc_dir_path=None,subject_list=None,session_list=None,
            clamp_thres=None,epoch_point_st=None,epoch_point_end=None,
            norm_per_sample=None,norm_per_electrode=None,norm_per_2sample_electrode=None,):

        
        self.preproc_dir_path = preproc_dir_path
        self.subject_list = subject_list
        self.session_list = session_list

        #getitem时做二次预处理
        self.clamp_thres = clamp_thres
        self.epoch_point_st = epoch_point_st
        self.epoch_point_end = epoch_point_end
        self.norm_per_sample = norm_per_sample#是否每个样本单独归一化
        self.norm_per_electrode = norm_per_electrode#是否分电极归一化
        self.norm_per_2sample_electrode = norm_per_2sample_electrode#是否每个样本单独分电极归一化

        self.eeg_path_dataset = eeg_path_dataset
        self.label_dataset = label_dataset
        self.imgNO_dataset = imgNO_dataset

        self.class_num = ori_class_num
        if eeg_path_dataset is not None:
            sample_num = len(eeg_path_dataset)
            self.label_onehot_dataset = np.zeros([sample_num,ori_class_num])
            for i in range(sample_num):
                classNO = label_dataset[i]-1
                self.label_onehot_dataset[i,classNO]=1
    
    def __len__(self):
        return len(self.eeg_path_dataset)

    def preprocess_twice_dataset(self, eeg_dataset):
        eeg_dataset = np.array(eeg_dataset)[:,self.epoch_point_st:self.epoch_point_end, :]
        # print('eeg_dataset:',eeg_dataset)
        # clamp
        if (not self.clamp_thres == 'none') and (not self.clamp_thres is None):
            eeg_dataset=clamp_per_electrode(eeg_dataset,self.clamp_thres)
            # print('eeg_dataset:',eeg_dataset)
        #归一化
        if self.norm_per_sample:
            eeg_dataset=normalize_per_sample(eeg_dataset)
        if self.norm_per_electrode:
            eeg_dataset=normalize_per_electrode(eeg_dataset)
        if self.norm_per_2sample_electrode:
            eeg_dataset=normalize_per_sample_electrode(eeg_dataset)
        return eeg_dataset

    def preprocess_twice(self, eeg):
        eeg = np.array(eeg)[self.epoch_point_st:self.epoch_point_end, :]
        # print('eeg:',eeg)
        # clamp
        if (not self.clamp_thres == 'none') and (not self.clamp_thres is None):
            eeg=(eeg-np.mean(eeg, axis=0))#必须先去均值，否则clamp可能偏差很大
            eeg[eeg >self.clamp_thres] = self.clamp_thres
            eeg[eeg < -self.clamp_thres] = -self.clamp_thres
            # print('eeg:',eeg)
        #归一化
        if self.norm_per_sample:
            eeg=(eeg-np.mean(eeg))/np.std(eeg)
            # print(eeg_npy_path,' np.std(eeg):',np.std(eeg))
        #归一化
        if self.norm_per_electrode:
            eeg= (eeg - np.mean(eeg, axis=0)) / np.std(eeg, axis=0)
        return eeg

    def __getitem__(self, idx):   
        #imgNO
        imgNO=self.imgNO_dataset[idx,:]
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        #label
        label_1_x=self.label_dataset[idx]
        label_onehot=np.zeros([self.class_num])
        label_onehot[label_1_x-1]=1
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        #eeg
        eeg_path=self.eeg_path_dataset[idx]
        eeg = np.load(os.path.join(self.preproc_dir_path,eeg_path))

        normed_eeg = self.preprocess_twice(eeg) #0524的并不是提前预处理好的了
        subNO=-99 #这回未记录subNO
        return (torch.tensor(normed_eeg).float(),
            torch.tensor(label_onehot).float(), 
            imgNO)
            # torch.tensor(imgNO).float(),     

    def split_dataset(self,split_proportion_list,seed):
        # 1、先设定好随机数
        split_num = len(split_proportion_list)
        sample_num, class_num = self.label_onehot_dataset.shape
        perm_shuffle = np.arange(sample_num)
        np.random.seed(seed)
        np.random.shuffle(perm_shuffle)
        # print(perm_shuffle)
        
        # --------------------------------------------------------------------
        # 2、计算每一类split后的样本数
        # sample_num_per_class = np.zeros([class_num])
        sample_num_per_class = np.sum(self.label_onehot_dataset,axis=0)
        _,sample_split_position = get_sample_num_per_class_splited(
            class_num, sample_num_per_class, split_proportion_list)
        # shape: class_num,split_num
        # --------------------------------------------------------------------
        # 3、原数据shuffle后，按类别sort，然后逐类切分
        label_0_x_dataset = np.argmax(self.label_onehot_dataset,axis=1).squeeze()
        label_0_x_dataset_shuffled = label_0_x_dataset[perm_shuffle]
        perm_sort = np.argsort(label_0_x_dataset_shuffled)
        
        perm_final = perm_shuffle[perm_sort]
        sampleNO_split_list = get_sampleNO_split(sample_split_position,perm_final,class_num,split_num)

        return sampleNO_split_list




# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
class My_Dataset_nopreload_split(My_Dataset_nopreload):
    def __init__(self,
                dataset_nopreload,
                sampleNO_split):
        super(My_Dataset_nopreload_split, self).__init__() #这里用默认值，来控制传入参数量的多少

        self.class_num = dataset_nopreload.class_num

        self.sampleNO_split = sampleNO_split

        self.preproc_dir_path = dataset_nopreload.preproc_dir_path
        self.subject_list = dataset_nopreload.subject_list
        self.session_list = dataset_nopreload.session_list

        #getitem时做二次预处理
        self.clamp_thres = dataset_nopreload.clamp_thres
        self.epoch_point_st = dataset_nopreload.epoch_point_st
        self.epoch_point_end = dataset_nopreload.epoch_point_end
        self.norm_per_sample = dataset_nopreload.norm_per_sample#是否每个样本单独归一化
        self.norm_per_electrode = dataset_nopreload.norm_per_electrode#是否分电极归一化
        self.norm_per_2sample_electrode = dataset_nopreload.norm_per_2sample_electrode#是否每个样本单独分电极归一化
        # SPECIAL~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        # 选出部分样本
        eeg_path_dataset = list(np.array(dataset_nopreload.eeg_path_dataset)[sampleNO_split])
        label_onehot_dataset = dataset_nopreload.label_onehot_dataset[sampleNO_split,:]
        imgNO_dataset = dataset_nopreload.imgNO_dataset[sampleNO_split,:]

        label_dataset = dataset_nopreload.label_dataset
        if label_dataset is not None:
            label_dataset = list(np.array(label_dataset)[sampleNO_split])
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        self.eeg_path_dataset = eeg_path_dataset
        self.label_onehot_dataset = label_onehot_dataset
        self.label_dataset = label_dataset
        self.imgNO_dataset = imgNO_dataset


class My_Dataset_nopreload_merge(My_Dataset_nopreload):
    def __init__(self,
                dataset_list):
        super(My_Dataset_nopreload_merge, self).__init__() #这里用默认值，来控制传入参数量的多少
        
        self.class_num = dataset_list[0].class_num

        self.preproc_dir_path = dataset_list[0].preproc_dir_path
        self.subject_list = dataset_list[0].subject_list
        self.session_list = dataset_list[0].session_list
        
        #getitem时做二次预处理
        self.clamp_thres = dataset_list[0].clamp_thres
        self.epoch_point_st = dataset_list[0].epoch_point_st
        self.epoch_point_end = dataset_list[0].epoch_point_end
        self.norm_per_sample = dataset_list[0].norm_per_sample#是否每个样本单独归一化
        self.norm_per_electrode = dataset_list[0].norm_per_electrode#是否分电极归一化
        self.norm_per_2sample_electrode = dataset_list[0].norm_per_2sample_electrode#是否每个样本单独分电极归一化
        # SPECIAL~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        # 融合所有样本
        label_onehot_dataset_merged = dataset_list[0].label_onehot_dataset
        eeg_path_dataset_merged = dataset_list[0].eeg_path_dataset
        label_dataset_merged = dataset_list[0].label_dataset
        imgNO_dataset_merged = dataset_list[0].imgNO_dataset
        # 
        subject_list_merged = dataset_list[0].subject_list
        session_list_merged = dataset_list[0].session_list
        
        for dataset_splited in dataset_list[1:]:
            label_onehot_dataset = dataset_splited.label_onehot_dataset
            eeg_path_dataset = dataset_splited.eeg_path_dataset
            label_dataset = dataset_splited.label_dataset
            imgNO_dataset = dataset_splited.imgNO_dataset

            #
            subject_list = dataset_splited.subject_list
            session_list = dataset_splited.session_list
            # - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~

            label_onehot_dataset_merged = np.concatenate([label_onehot_dataset_merged, label_onehot_dataset],axis=0)
            imgNO_dataset_merged = np.concatenate([imgNO_dataset_merged, imgNO_dataset],axis=0)
            if (eeg_path_dataset is not None):
                eeg_path_dataset_merged = eeg_path_dataset_merged+eeg_path_dataset
            if (label_dataset is not None):
                label_dataset_merged = label_dataset_merged+label_dataset
            #
            if (subject_list is not None):
                subject_list_merged = subject_list_merged+subject_list
            #有一个session为none，那么就全置为none
            # （此时不代表使用了所有session，注意在外面的代码，
            # 除了只有1个session的情况，不要把session-list置为None）
            if session_list_merged is not None:
                if session_list is None:
                    session_list_merged=None
                else:
                    session_list_merged = session_list_merged+session_list
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        self.label_onehot_dataset = label_onehot_dataset_merged
        self.eeg_path_dataset = eeg_path_dataset_merged
        self.label_dataset = label_dataset_merged
        self.imgNO_dataset = imgNO_dataset_merged
        #
        self.subject_list = subject_list_merged
        self.session_list = session_list_merged

    def divide(self,sample_num_per_class):

        # 1、先设定好随机数
        seed = 2025
        # split_num = len(split_proportion_list)
        sample_num, class_num = self.label_onehot_dataset.shape
        perm_shuffle = np.arange(sample_num)
        np.random.seed(seed)
        np.random.shuffle(perm_shuffle)
        # print(perm_shuffle)
        
        # # --------------------------------------------------------------------
        # 3、原数据shuffle后，按类别sort，然后逐类切分
        label_0_x_dataset = np.argmax(self.label_onehot_dataset,axis=1).squeeze()
        label_0_x_dataset_shuffled = label_0_x_dataset[perm_shuffle]
        perm_sort = np.argsort(label_0_x_dataset_shuffled)
        
        perm_shuffle_per_class = perm_shuffle[perm_sort]

            
        sampleNO_divide=[]
        sampleNO_st = 0
        for classNO in range(class_num):
            sample_num_this_class = np.sum(label_0_x_dataset_shuffled==classNO)

            if sample_num_per_class>sample_num_this_class:
                assert False,'sample_num_per_class 不能大于 sample_num_this_class！'

            st_idx = sampleNO_st
            end_idx = st_idx + sample_num_per_class

            #注意这里不能用+=，会出现错误结果，导致整体都+了
            sampleNO_divide=sampleNO_divide + perm_shuffle_per_class[st_idx:end_idx].tolist()

            sampleNO_st+=sample_num_this_class

        # SPECIAL~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        # 选出部分样本
        # self.eeg_dataset = self.eeg_dataset[sampleNO_divide,:,:]
        self.label_onehot_dataset = self.label_onehot_dataset[sampleNO_divide,:]
        self.imgNO_dataset = self.imgNO_dataset[sampleNO_divide,:]
        
        if self.eeg_path_dataset is not None:
            self.eeg_path_dataset = list(np.array(self.eeg_path_dataset)[sampleNO_divide])
            
        if self.label_dataset is not None:
            self.label_dataset = list(np.array(self.label_dataset)[sampleNO_divide])
        
        print('origin sample_num=',sample_num,'divided to ',self.label_onehot_dataset.shape[0])


# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 






