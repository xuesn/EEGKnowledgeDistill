import os
import random
import numpy as np
import torch
import logging

from PIL import Image

def get_a_img(img_path):
    img = Image.open(img_path)
    img_width, img_height = 224,224
    img_arr = np.array(img.resize([img_width, img_height]))
    # 灰度图像转RGB就是把通道复制3遍
    if len(img_arr.shape) == 2:
        img_arr = np.expand_dims(img_arr,2).repeat(3,axis=2)
    return img_arr
    

class My_Dataset_nopreload(torch.utils.data.Dataset):

    #主构造函数
    def __init__(self,
            img_path_list,):

        
        self.img_path_list = img_path_list
    
    def __len__(self):
        return len(self.img_path_list)


    def __getitem__(self, idx):   
        img_path=self.img_path_list[idx]
   
        img_arr = get_a_img(img_path)

        return torch.tensor(img_arr).float()






