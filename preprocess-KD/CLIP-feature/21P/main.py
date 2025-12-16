




import os
cudaNO_list = [ 0 ]
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


# 1、装载图片名列表
# 4w张图片的编号顺序
img_dir_path = "/share/eeg_datasets/Vision/21_Purdue/CVPR2021-02785/stimuli/"
img_list = os.listdir(img_dir_path)
img_list.sort()

# 2、构建图片类别标签（训练linear用）
img_num = 40000
class_num = 40
label_onehot_40000 = np.zeros([img_num, class_num],np.int8)
for i in range(img_num):
    classNO = int(i/1000)
    label_onehot_40000[i,classNO] = 1


# 3、读取图片
# preprocessed_img_save_path='/data/snxue/eeg_preprocess/21Purdue-img/'+'21Purdue_normed_img_arr.npy'
preprocessed_img_save_path='/data/snxue/eeg_preprocess/21Purdue-img/'+'21Purdue_raw_img_arr.npy'
'''
img_set=np.load(preprocessed_img_save_path)
print('img_set',' loaded!')
'''

# 设置放缩后图像大小
img_height = 224
img_width = 224
RGBnum = 3
img_set = np.zeros([img_num, img_width, img_height, RGBnum])
# 网上找的ImageNet 均值、标准差
img_mean= [0.485, 0.456, 0.406]
img_std= [0.229, 0.224, 0.225]
# 路径
for imgNO, img_str in enumerate(img_list):
    img_path =os.path.join(img_dir_path,img_str)
    img = Image.open(img_path)
    img_arr = np.array(img.resize([img_width, img_height]))
    # 灰度图像转RGB就是把通道复制3遍
    if len(img_arr.shape) == 2:
        img_arr = np.expand_dims(img_arr,2).repeat(3,axis=2)
    # img_set[imgNO, :, :, :] = (img_arr/255-img_mean)/img_std # 2025-0310
    img_set[imgNO, :, :, :] = img_arr
    print('img',imgNO,' loaded!')

np.save(preprocessed_img_save_path,img_set.astype(np.float32))
print('!!!'+preprocessed_img_save_path+' saved!!!')
#6G~5742.1875MB = 10000张 * 3*224*224（3*50176）  /1024/1024


# 

# 4、提取特征
tensor_img_set = torch.tensor(img_set, dtype=torch.float32)
# dataloader
batch_size = 512
batch_size = 40
start_time = time.time()
encode_batch_size = batch_size
# tensor_img_set = torch.tensor(img_set, dtype=torch.float32)
# if (model_str == 'alex') | (model_str == 'vit_b_16') | (model_str == 'resnet18') | (model_str == 'resnet34') | (model_str == 'resnet50') | (model_str == 'resnet101'):
#     # vit/resnet/clip的输入要为 n, c, h, w = x.shape
#     tensor_img_set = tensor_img_set.permute(0, 3, 2, 1)
encode_set = torch.utils.data.TensorDataset(
    tensor_img_set, )    
encode_loader = torch.utils.data.DataLoader(encode_set, encode_batch_size,
                                            shuffle=False,)
# shuffle=False可一定要注意啊，这里如果打乱了，label就全乱了
print('dataloader constructed!',time.time()-start_time)


# *******************************************************************
# model装载模型
from transformers import CLIPProcessor, CLIPModel

# clip_vit_base_path =  '/data/snxue/clip-vit-base-patch32'
# model_base = CLIPModel.from_pretrained(clip_vit_base_path, local_files_only=True)
# processor_base = CLIPProcessor.from_pretrained(clip_vit_base_path, local_files_only=True)

# clip_vit_large_path = '/data/snxue/clip-vit-large-patch14'
# model_large = CLIPModel.from_pretrained(clip_vit_large_path, local_files_only=True)
# processor_large = CLIPProcessor.from_pretrained(clip_vit_large_path, local_files_only=True)

finished_num = 0
model_str_list = ['clip-vit-base-patch32', 'clip-vit-large-patch14', ]
for model_str in model_str_list[finished_num:]:
    start_time = time.time()
    # 选择模型----------------------------------------------------------------------------------------------------
    # 只提特征，去分类头
    if model_str == 'clip-vit-base-patch32':
        clip_vit_path =  '/data/snxue/clip-vit-base-patch32'
    elif model_str == 'clip-vit-large-patch14':
        clip_vit_path = '/data/snxue/clip-vit-large-patch14'
    else:
        assert 1 != 1, '其他模型没做哦~~~'
    model = CLIPModel.from_pretrained(clip_vit_path, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(clip_vit_path, local_files_only=True)
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
        for step, (batch_x,) in enumerate(encode_loader):  # 要写成(batch_x,)，不能(batch_x)哦
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
    feat_save_path = '/data/snxue/visual_embedding/21Purdue/'
    feat_flie_name = model_str+'_dim'+str(img_feat_dim)+'.npy'
    feat_file_fullpath=os.path.join(feat_save_path,feat_flie_name)
    np.save(feat_file_fullpath,sample_feature)
    print(model_str,'特征维度为',img_feat_dim,'。而sample_feature中有',np.sum(sample_feature==0),'个零值。样本数为',sample_feature.shape[0])
    print('!!!', feat_file_fullpath, ' saved!!!',time.time()-start_time)


