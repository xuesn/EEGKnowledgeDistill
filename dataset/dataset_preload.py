import os
import random
import numpy as np
import torch
import logging

from .utils_dataset import *

class My_Dataset_json_preload_classMerge(torch.utils.data.Dataset):
    #主构造函数
    def __init__(self,
                eeg_json_path=None, label_json_path=None, imgNO_json_path=None,
                ori_timepoint_num=None,ori_electrode_num=None,ori_class_num=None,
                preproc_dir_path=None,subject_list=None,session_list=None,
                clamp_thres=None,epoch_point_st=None,epoch_point_end=None,
                norm_per_sample=None,norm_per_electrode=None,norm_per_2sample_electrode=None,):
        # 记录一下该数据集的数据来源
        self.eeg_json_path = eeg_json_path
        self.label_json_path = label_json_path
        self.imgNO_json_path = imgNO_json_path

        
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


        if eeg_json_path is None or label_json_path is None or imgNO_json_path is None:
            self.eeg_dataset = None
            self.label_onehot_dataset = None
            self.label_dataset = None
            self.imgNO_dataset = None
        else:
            (eeg_dataset,label_onehot_dataset,         
            eeg_path_dataset,label_dataset,imgNO_dataset)=get_eeg_label_dataset_from_json( 
                None,
                eeg_json_path, label_json_path, imgNO_json_path, preproc_dir_path,
                subject_list,session_list,
                ori_timepoint_num,ori_electrode_num,ori_class_num,)
            

            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            # 二次预处理
            normed_eeg_dataset = self.preprocess_twice_dataset(eeg_dataset)
            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            self.eeg_dataset = normed_eeg_dataset
            self.label_onehot_dataset = label_onehot_dataset
            self.eeg_path_dataset = eeg_path_dataset
            self.label_dataset = label_dataset
            self.imgNO_dataset = imgNO_dataset
    
    def __len__(self):
        return self.eeg_dataset.shape[0]

    def __getitem__(self, idx):   
        #imgNO
        imgNO=self.imgNO_dataset[idx,:]
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        #label
        label_onehot=self.label_onehot_dataset[idx,:]
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        #eeg
        eeg=self.eeg_dataset[idx,:,:]
        normed_eeg = eeg #提前预处理好了

        subNO=-99 #这回未记录subNO
        return (torch.tensor(normed_eeg).float(),
            torch.tensor(label_onehot).float(),
            imgNO,)
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
        # --------------------------------------------------------------------
        # 4、对eeg/label/time/acq切分 放在外面做    

        
        '''
        # # --------------------------------------------------------------------
        # 随机划分：
        sample_num = self.label_onehot_dataset.shape[0]   

        # 测试用
        # import numpy as np
        # split_proportion_list=[1]*10
        # sample_num = 100

        perm = np.arange(sample_num)
        np.random.shuffle(perm)
        split_num = len(split_proportion_list)
        sample_num_splited = np.zeros(split_num)        
        denominator=sum(split_proportion_list)
        for splitNO,numerator in enumerate(split_proportion_list[:-1]):
            sample_num_this_split = np.floor(sample_num * numerator / denominator)  #用floor，避免取的样本数超出界限
            sample_num_splited[splitNO] = sample_num_this_split
        sample_num_splited[split_num-1] = sample_num - np.sum(sample_num_splited)
        sample_split_position = np.add.accumulate(sample_num_splited)
        sample_split_position = np.insert(sample_split_position,0,0).astype(np.int32)
        sampleNO_split_list=[[]]*split_num
        for splitNO in range(split_num):
            stNO = splitNO
            endNO= stNO + 1
            st_idx = sample_split_position[stNO]
            end_idx= sample_split_position[endNO]

            #注意这里不能用+=，会出现错误结果，导致整体都+了
            sampleNO_split_list[splitNO]=sampleNO_split_list[splitNO] + perm[st_idx:end_idx].tolist()
        '''

        return sampleNO_split_list

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

# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
class My_Dataset_preload_split(My_Dataset_json_preload_classMerge):
    def __init__(self,
                dataset_preload,
                sampleNO_split):
        super(My_Dataset_preload_split, self).__init__() #这里用默认值，来控制传入参数量的多少
        self.sampleNO_split = sampleNO_split
        
        # 记录一下该数据集的数据来源
        self.eeg_json_path = dataset_preload.eeg_json_path
        self.label_json_path = dataset_preload.label_json_path
        self.imgNO_json_path = dataset_preload.imgNO_json_path


        self.preproc_dir_path = dataset_preload.preproc_dir_path
        self.subject_list = dataset_preload.subject_list
        self.session_list = dataset_preload.session_list

        #getitem时做二次预处理
        self.clamp_thres = dataset_preload.clamp_thres
        self.epoch_point_st = dataset_preload.epoch_point_st
        self.epoch_point_end = dataset_preload.epoch_point_end
        self.norm_per_sample = dataset_preload.norm_per_sample#是否每个样本单独归一化
        self.norm_per_electrode = dataset_preload.norm_per_electrode#是否分电极归一化
        self.norm_per_2sample_electrode = dataset_preload.norm_per_2sample_electrode#是否每个样本单独分电极归一化
        # SPECIAL~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        # 选出部分样本
        eeg_dataset = dataset_preload.eeg_dataset[sampleNO_split,:,:]
        label_onehot_dataset = dataset_preload.label_onehot_dataset[sampleNO_split,:]
        imgNO_dataset = dataset_preload.imgNO_dataset[sampleNO_split,:]
        eeg_path_dataset = dataset_preload.eeg_path_dataset
        if eeg_path_dataset is not None:
            eeg_path_dataset = list(np.array(eeg_path_dataset)[sampleNO_split])
        label_dataset = dataset_preload.label_dataset
        if label_dataset is not None:
            label_dataset = list(np.array(label_dataset)[sampleNO_split])
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        self.eeg_dataset = eeg_dataset
        self.label_onehot_dataset = label_onehot_dataset
        self.eeg_path_dataset = eeg_path_dataset
        self.label_dataset = label_dataset
        self.imgNO_dataset = imgNO_dataset


class My_Dataset_preload_merge(My_Dataset_json_preload_classMerge):
    def __init__(self,
                dataset_list):
        super(My_Dataset_preload_merge, self).__init__() #这里用默认值，来控制传入参数量的多少
        
        # 记录一下该数据集的数据来源
        self.eeg_json_path = dataset_list[0].eeg_json_path
        self.label_json_path = dataset_list[0].label_json_path
        self.imgNO_json_path = dataset_list[0].imgNO_json_path


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
        eeg_dataset_merged = dataset_list[0].eeg_dataset
        label_onehot_dataset_merged = dataset_list[0].label_onehot_dataset
        eeg_path_dataset_merged = dataset_list[0].eeg_path_dataset
        label_dataset_merged = dataset_list[0].label_dataset
        imgNO_dataset_merged = dataset_list[0].imgNO_dataset
        # 
        eeg_json_path_merged = dataset_list[0].eeg_json_path
        label_json_path_merged = dataset_list[0].label_json_path
        imgNO_json_path_merged = dataset_list[0].imgNO_json_path
        subject_list_merged = dataset_list[0].subject_list
        session_list_merged = dataset_list[0].session_list
        
        for dataset_splited in dataset_list[1:]:
            eeg_dataset = dataset_splited.eeg_dataset
            label_onehot_dataset = dataset_splited.label_onehot_dataset
            eeg_path_dataset = dataset_splited.eeg_path_dataset
            label_dataset = dataset_splited.label_dataset
            imgNO_dataset = dataset_splited.imgNO_dataset

            #
            eeg_json_path = dataset_splited.eeg_json_path
            label_json_path = dataset_splited.label_json_path
            imgNO_json_path = dataset_splited.imgNO_json_path
            subject_list = dataset_splited.subject_list
            session_list = dataset_splited.session_list
            # - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
            eeg_dataset_merged = np.concatenate([eeg_dataset_merged, eeg_dataset],axis=0)
            label_onehot_dataset_merged = np.concatenate([label_onehot_dataset_merged, label_onehot_dataset],axis=0)
            imgNO_dataset_merged = np.concatenate([imgNO_dataset_merged, imgNO_dataset],axis=0)
            if (eeg_path_dataset is not None):
                eeg_path_dataset_merged = eeg_path_dataset_merged+eeg_path_dataset
            if (label_dataset is not None):
                label_dataset_merged = label_dataset_merged+label_dataset
            #
            # eeg_json_path_merged = np.concatenate([eeg_json_path_merged, eeg_json_path],axis=0)
            # label_json_path_merged = np.concatenate([label_json_path_merged, label_json_path],axis=0)
            # subject_list_merged = np.concatenate([subject_list_merged, subject_list],axis=0)
            # session_list_merged = np.concatenate([session_list_merged, session_list],axis=0)
            if (eeg_json_path is not None):
                eeg_json_path_merged = eeg_json_path_merged+eeg_json_path
            if (label_json_path is not None):
                label_json_path_merged = label_json_path_merged+label_json_path
            if (imgNO_json_path is not None):
                imgNO_json_path_merged = imgNO_json_path_merged+imgNO_json_path
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
        self.eeg_dataset = eeg_dataset_merged
        self.label_onehot_dataset = label_onehot_dataset_merged
        self.eeg_path_dataset = eeg_path_dataset_merged
        self.label_dataset = label_dataset_merged
        self.imgNO_dataset = imgNO_dataset_merged
        #
        self.eeg_json_path = eeg_json_path_merged
        self.label_json_path = label_json_path_merged
        self.imgNO_json_path = imgNO_json_path_merged
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
        self.eeg_dataset = self.eeg_dataset[sampleNO_divide,:,:]
        self.label_onehot_dataset = self.label_onehot_dataset[sampleNO_divide,:]
        self.imgNO_dataset = self.imgNO_dataset[sampleNO_divide,:]
        
        if self.eeg_path_dataset is not None:
            self.eeg_path_dataset = list(np.array(self.eeg_path_dataset)[sampleNO_divide])
            
        if self.label_dataset is not None:
            self.label_dataset = list(np.array(self.label_dataset)[sampleNO_divide])
        
        print('origin sample_num=',sample_num,'divided to ',self.label_onehot_dataset.shape[0])


# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 


class My_Dataset_array_preload(My_Dataset_json_preload_classMerge):
    #主构造函数
    def __init__(self,
                eeg_dataset=None,label_dataset=None,imgNO_dataset=None,
                eeg_npy_dir=None, label_npy_dir=None,imgNO_npy_dir=None,
                ori_timepoint_num=None,ori_electrode_num=None,ori_class_num=None,
                preproc_dir_path=None,subject_list=None,session_list=None,
                clamp_thres=None,epoch_point_st=None,epoch_point_end=None,
                norm_per_sample=None,norm_per_electrode=None,norm_per_2sample_electrode=None,):
        # 记录一下该数据集的数据来源
        self.eeg_npy_dir = eeg_npy_dir
        self.label_npy_dir = label_npy_dir
        self.imgNO_npy_dir = imgNO_npy_dir
        self.eeg_json_path = None
        self.label_json_path = None
        self.imgNO_json_path = None

        
        self.preproc_dir_path = preproc_dir_path
        self.subject_list = subject_list
        self.session_list = None #我决定在npy-load中删去session_list,因为有的数据集没有session，不好统一命名

        #getitem时做二次预处理
        self.clamp_thres = clamp_thres
        self.epoch_point_st = epoch_point_st
        self.epoch_point_end = epoch_point_end
        self.norm_per_sample = norm_per_sample#是否每个样本单独归一化
        self.norm_per_electrode = norm_per_electrode#是否分电极归一化
        self.norm_per_2sample_electrode = norm_per_2sample_electrode#是否每个样本单独分电极归一化

        self.eeg_path_dataset = None
        if eeg_dataset is None or label_dataset is None or imgNO_dataset is None:
            self.eeg_dataset = None
            self.label_onehot_dataset = None
            self.label_dataset = None
            self.imgNO_dataset = None
        else:    
            # 这个函数只是做了转list、选部分类、转one-hot
            (eeg_dataset,label_onehot_dataset,
                label_dataset,imgNO_dataset) = get_eeg_label_dataset_from_array( 
                    eeg_dataset, label_dataset, imgNO_dataset,
                    ori_timepoint_num,ori_electrode_num,ori_class_num,)
            

            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            # 二次预处理
            normed_eeg_dataset = self.preprocess_twice_dataset(eeg_dataset)
            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            self.eeg_dataset = normed_eeg_dataset
            self.label_onehot_dataset = label_onehot_dataset
            self.label_dataset = label_dataset
            self.imgNO_dataset = imgNO_dataset


