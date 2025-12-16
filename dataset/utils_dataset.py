
import os
import random
import numpy as np
import torch
import logging




def get_eeg_label_dataset_from_array( 
        eeg_dataset, label_dataset, imgNO_dataset,
        ori_timepoint_num,ori_electrode_num,ori_class_num,):
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # 转one-hot
    sample_num = len(label_dataset)
    label_onehot_dataset = np.zeros([sample_num,ori_class_num])      
    # 法1
    label_onehot_dataset[np.arange(label_dataset.size), (label_dataset-1).reshape([-1, ])] = 1
    # # 法2
    # for sampleNO,label in enumerate(label_dataset):
    #     label_onehot_dataset[sampleNO,label]=1
        
    label_dataset = label_dataset.tolist()  #按传统，把label转为list
    return eeg_dataset,label_onehot_dataset,label_dataset,imgNO_dataset




# split - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
def get_sample_num_per_class_splited(class_num, sample_num_per_class, split_proportion_list):
    split_num = len(split_proportion_list)
    denominator=sum(split_proportion_list)
    # 获取每一类split后的样本数
    sample_num_per_class_splited = np.zeros([class_num,split_num])
    for classNO in range(class_num):
        for splitNO in range(split_num-1):
            numerator = split_proportion_list[splitNO]
            sample_num_this_class = sample_num_per_class[classNO]
            sample_num_this_class_this_split = np.floor(sample_num_this_class * numerator / denominator)  #用floor，避免取的样本数超出界限
            sample_num_per_class_splited[classNO,splitNO] = sample_num_this_class_this_split
    # 最后一个split，取剩下的所有样本
    sample_num_per_class_splited[:,split_num-1] = sample_num_per_class - np.sum(sample_num_per_class_splited,axis=1)
    
    '''测试代码
    import numpy as np
    sample_num_per_class_splited = np.random.randint(0,10,[6,3])
    '''
    sample_num_class_split = sample_num_per_class_splited.reshape(1,-1).squeeze() #(-1,1) 也一样
    sample_split_position = np.add.accumulate(sample_num_class_split)

    return sample_num_per_class_splited,sample_split_position

def get_sampleNO_split(sample_split_position,perm_final,class_num,split_num):
    sample_split_position = np.insert(sample_split_position,0,0).astype(np.int32)
    sampleNO_split=[[]]*split_num
    for classNO in range(class_num):
        for splitNO in range(split_num):
            stNO = classNO*split_num + splitNO
            endNO= stNO + 1
            st_idx = sample_split_position[stNO]
            end_idx= sample_split_position[endNO]

            #注意这里不能用+=，会出现错误结果，导致整体都+了
            sampleNO_split[splitNO]=sampleNO_split[splitNO] + perm_final[st_idx:end_idx].tolist()
            # print(perm_final[st_idx:end_idx].tolist())
            # print(splitNO,classNO,sampleNO_split[splitNO], perm_final[st_idx:end_idx].tolist())

            # print(classNO,splitNO,st_idx,end_idx)
    return sampleNO_split


# 预处理 - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~  
def clamp_per_electrode(sample_time_electrode,clamp_thres):

    #必须先去均值，否则clamp可能偏差很大
    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    sample_num, time_num, electrode_num = sample_time_electrode.shape
    time_sample_electrode = sample_time_electrode.transpose(1,0,2)

    time_sampleElectrode = time_sample_electrode.reshape(
        [time_num, -1]) 
    time_sampleElectrode = (time_sampleElectrode - np.mean(time_sampleElectrode, axis=0)) 
    time_sample_electrode = time_sampleElectrode.reshape(
        [time_num, sample_num, electrode_num])
    sample_time_electrode = time_sample_electrode.transpose(1,0,2)
    
    sample_time_electrode[sample_time_electrode > clamp_thres] = clamp_thres
    sample_time_electrode[sample_time_electrode < -clamp_thres] = -clamp_thres

    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    # print('sample_time_electrode-shape:', sample_time_electrode.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(sample_time_electrode[1,:,:])

    return sample_time_electrode

def normalize_per_sample( sample_time_electrode):
    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    # 这里统一归一化一下吧  其实已经切分过，切分前归一化更合理吗？
    # 切分后归一化也有好处，因为切分前的记录了休息时间乱动的脑电
    sample_num, time_num, electrode_num = sample_time_electrode.shape
    sample_timeElectrode = sample_time_electrode.reshape(
        [sample_num,-1])  # 参考其他文章，逐电极归一化，而不是单样本归一化（之前做过，后面也可以重新试试）
    sample_timeElectrode = (sample_timeElectrode - np.mean(
        sample_timeElectrode, axis=0)) / np.std(sample_timeElectrode, axis=0)
    sample_time_electrode = sample_timeElectrode.reshape(
        [sample_num, time_num, electrode_num])
    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    print('sample_time_electrode-shape:', sample_time_electrode.shape)
    return sample_time_electrode

def normalize_per_electrode( sample_time_electrode):
    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    # 这里统一归一化一下吧  其实已经切分过，切分前归一化更合理吗？
    # 切分后归一化也有好处，因为切分前的记录了休息时间乱动的脑电
    sample_num, time_num, electrode_num = sample_time_electrode.shape
    sampleTime_electrode = sample_time_electrode.reshape(
        [-1, electrode_num])  # 参考其他文章，逐电极归一化，而不是单样本归一化（之前做过，后面也可以重新试试）
    sampleTime_electrode = (sampleTime_electrode - np.mean(
        sampleTime_electrode, axis=0)) / np.std(sampleTime_electrode, axis=0)
    sample_time_electrode = sampleTime_electrode.reshape(
        [sample_num, time_num, electrode_num])
    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    print('sample_time_electrode-shape:', sample_time_electrode.shape)
    return sample_time_electrode

def normalize_per_sample_electrode( sample_time_electrode):
    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    sample_num, time_num, electrode_num = sample_time_electrode.shape
    time_sample_electrode = sample_time_electrode.transpose(1,0,2)

    time_sampleElectrode = time_sample_electrode.reshape(
        [time_num, -1]) 
    mean_ = np.mean(time_sampleElectrode, axis=0)
    std_ = np.std(time_sampleElectrode, axis=0)
    time_sampleElectrode = (time_sampleElectrode - mean_) / std_
    time_sample_electrode = time_sampleElectrode.reshape(
        [time_num, sample_num, electrode_num])
    
    # np.sum(np.isnan(time_sampleElectrode))
    # np.sum(std_==0)  # # 一定要检查有无为0的数据
    # np.argwhere(std_ ==0)

    sample_time_electrode = time_sample_electrode.transpose(1,0,2)
    # print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    # print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    # print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    # print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    # print('sample_time_electrode-shape:', sample_time_electrode.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(sample_time_electrode[1,:,:])
    return sample_time_electrode



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
def sample_dir_sort_criteria(sample_dir):
    #返回文件夹名中的尾部样本数
    return int(sample_dir.split('_')[-1])

def sample_json_npy_sort_criteria(sample_file):
    #返回文件夹名中的尾部样本数
    return int(sample_file.split('.')[0].split('sample')[-1])