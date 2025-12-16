




import os
cudaNO_list = [ 4 ]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
    map(str, cudaNO_list))  # 一般在程序开头设置

import json
import numpy as np

import time
import pandas as pd

import torch
import torchvision.models as models
from torchvision.transforms import Resize
import torchvision.transforms as transforms
from PIL import Image
# import cv2

import torch.nn as nn
import random
from torch import optim

cudaNO = '0'
device = torch.device(
    "cuda:"+cudaNO if torch.cuda.is_available() else "cpu")


# 1、图片路径列表  # 2、并构建图片类别标签（训练linear用）
train_img_dir = '/share/eeg_datasets/Vision/22_Germany/Image-set/training_images/'
test_img_dir = '/share/eeg_datasets/Vision/22_Germany/Image-set/test_images/'
# train test 统一编号，test跟在train后面
img_path_list=[]



img_num = 1654*10+200*1
class_num = 1654+200
label_onehot = np.zeros([img_num, class_num],np.int32)
classNO_current = 0
index_st = 0
index_class_str_list = []

def get_img_path_list_and_label_onehot(train_or_test_img_dir,img_path_list,index_st,label_onehot,classNO_current,index_class_str_list):
    img_dir_list = os.listdir(train_or_test_img_dir)
    img_dir_list.sort()

    index_class_str_list+=img_dir_list

    for img_dir in img_dir_list:
        img_dir_path = os.path.join(train_or_test_img_dir,img_dir)    
        img_list = os.listdir(img_dir_path)
        img_list.sort()
        img_path_list+=[os.path.join(img_dir_path,img_str)  for img_str in img_list]

        # 每一个dir都是一个类别
        index_end = index_st + len(img_list)
        label_onehot[index_st:index_end,classNO_current] = 1
        index_st = index_end
        classNO_current+=1

    return img_path_list,index_st,label_onehot,classNO_current,index_class_str_list

img_path_list,index_st,label_onehot,classNO_current,index_class_str_list = get_img_path_list_and_label_onehot(train_img_dir,img_path_list,index_st,label_onehot,classNO_current,index_class_str_list)

img_path_list,index_st,label_onehot,classNO_current,index_class_str_list = get_img_path_list_and_label_onehot(test_img_dir,img_path_list,index_st,label_onehot,classNO_current,index_class_str_list)

print(len(img_path_list))
print(index_st,classNO_current)
print(len(index_class_str_list))

print(label_onehot)

print(index_class_str_list)
class_str_list = ['_'.join(index_class_str.split('_')[1:]) for index_class_str in index_class_str_list]
print(class_str_list)

img_num = len(img_path_list)


#SPECIAL
# 4、提取特征

# dataloader
model_str_list = ['clip-vit-base-patch32',  ]
batch_size = 40*25
model_str_list = [ 'clip-vit-large-patch14', ]
batch_size = 40*10
batch_size = 40*5
start_time = time.time()
encode_batch_size = batch_size

# if (model_str == 'alex') | (model_str == 'vit_b_16') | (model_str == 'resnet18') | (model_str == 'resnet34') | (model_str == 'resnet50') | (model_str == 'resnet101'):
#     # vit/resnet/clip的输入要为 n, c, h, w = x.shape
#     tensor_img_set = tensor_img_set.permute(0, 3, 2, 1)

from dataset_img2 import My_Dataset_nopreload
encode_set = My_Dataset_nopreload(img_path_list)
encode_loader = torch.utils.data.DataLoader(encode_set, encode_batch_size, shuffle=False)
# shuffle=False可一定要注意啊，这里如果打乱了，label就全乱了
print('dataloader constructed!',time.time()-start_time)


#SPECIAL


# *******************************************************************
# model装载模型
from transformers import CLIPProcessor, CLIPModel

clip_vit_base_path =  '/data/snxue/clip-vit-base-patch32'
model_base = CLIPModel.from_pretrained(clip_vit_base_path, local_files_only=True)
processor_base = CLIPProcessor.from_pretrained(clip_vit_base_path, local_files_only=True)

clip_vit_large_path = '/data/snxue/clip-vit-large-patch14'
model_large = CLIPModel.from_pretrained(clip_vit_large_path, local_files_only=True)
processor_large = CLIPProcessor.from_pretrained(clip_vit_large_path, local_files_only=True)

finished_num = 0
for model_str in model_str_list[finished_num:]:
    start_time = time.time()
    # 选择模型----------------------------------------------------------------------------------------------------
    # 只提特征，去分类头
    if model_str == 'clip-vit-base-patch32':
        clip_vit_path =  '/data/snxue/clip-vit-base-patch32'
        model = model_base
        processor = processor_base
    elif model_str == 'clip-vit-large-patch14':
        clip_vit_path = '/data/snxue/clip-vit-large-patch14'
        model = model_large
        processor = processor_large
    else:
        assert 1 != 1, '其他模型没做哦~~~'
    # # 1、proj后的特征
    # img_feat_dim = model.visual_projection.out_features 
    # 2、proj前的特征
    img_feat_dim = model.visual_projection.in_features 
    # model.visual_projection = torch.nn.Sequential()
    # 这样不行 model.visual_projection = torch.nn.Sequential()；2025 0311 直接去改了源码 "/data/snxue/miniconda3/envs/torch-a6000/lib/python3.12/site-packages/transformers/models/clip/modeling_clip.py"
    
    print(model_str,'loaded!',time.time()-start_time)
    print('img_feat_dim:',img_feat_dim)



    # 模型提取特征----------------------------------------------------------------------------------------------------
    # 提取原始脑电数据的特征
    # model_enc_proj = new_model  # 预装载好权重的enc模型  new_model太麻烦 不如直接把分类头改为sequence或dropout
    model_enc_proj = model  # 预装载好权重的enc模型
    model_enc_proj.to(device)
    sample_feature = np.zeros([img_num, img_feat_dim])

    # else:

    # 开始提取----------------------------------------------------------------------------------------------------
    model_enc_proj.eval()
    with torch.no_grad():  # 注意一定要加 不然会保存梯度 占大量GPU
        # for step, (batch_x,) in enumerate(encode_loader):  # 要写成(batch_x,)，不能(batch_x)哦
        for step, batch_x in enumerate(encode_loader):  # 要写成(batch_x,)，不能(batch_x)哦
            # break
            batch_x = batch_x.to(device)  # bs=50 大概占800
            # out = model_enc_proj(batch_x)  # bs=50 大概又占800

            #0310 
            # 数据预处理：在加载模型后，需要对输入图像进行预处理。可以使用CLIPProcessor对图像和文本进行处理，生成模型所需的输入格式：
            inputs = processor(text=["a photo of a cat", ], images=batch_x, return_tensors="pt", padding=True)
            inputs = inputs.to(device)  # bs=50 大概占800
            
            # 模型推理：将预处理后的数据输入到模型中，进行推理：
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds  # image_embeds.shape: 512/768
            print(image_embeds.shape)
            out = image_embeds

            if (step+1)*encode_batch_size < img_num:
                sample_feature[step * encode_batch_size:
                                (step+1) * encode_batch_size, :] = out.cpu().detach().numpy()
            else:
                sample_feature[step*encode_batch_size:,
                                :] = out.cpu().detach().numpy()
            print('step',step,'finished!')


    # 保存特征----------------------------------------------------------------------------------------------------
    # 每个特征保存为1个npy
    feat_save_path = '/data/snxue/visual_embedding/22Germany/'
    feat_flie_name = model_str+'_dim'+str(img_feat_dim)+'.npy'
    feat_file_fullpath=os.path.join(feat_save_path,feat_flie_name)
    np.save(feat_file_fullpath,sample_feature)
    print(model_str,'特征维度为',img_feat_dim,'。而sample_feature中有',np.sum(sample_feature==0),'个零值。样本数为',sample_feature.shape[0])
    print('!!!', feat_file_fullpath, ' saved!!!',time.time()-start_time)


