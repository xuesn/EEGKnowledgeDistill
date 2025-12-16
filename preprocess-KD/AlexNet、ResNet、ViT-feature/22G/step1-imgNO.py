import os
import json
import numpy as np


# 0、构建img_str与imgNO的对应表 (     个图片名构成的列表，其位置就是imgNO号)

train_img_dir = '/share/eeg_datasets/Vision/22_Germany/Image-set/training_images/'
test_img_dir = '/share/eeg_datasets/Vision/22_Germany/Image-set/test_images/'
# train test 统一编号，test跟在train后面
img_str_list=[]


img_dir_list = os.listdir(train_img_dir)
img_dir_list.sort()
for img_dir in img_dir_list:
    img_dir_path = os.path.join(train_img_dir,img_dir)    
    img_list = os.listdir(img_dir_path)
    img_list.sort()
    img_str_list+=img_list
img_dir_list = os.listdir(test_img_dir)
img_dir_list.sort()
for img_dir in img_dir_list:
    img_dir_path = os.path.join(test_img_dir,img_dir)    
    img_list = os.listdir(img_dir_path)
    img_list.sort()
    img_str_list+=img_list
# 16740==len(img_str_list)  1654*10+200*1

overwrite_flag=1
save_dir = '/data/snxue/eeg_preprocess/22Germany-img/'
img_str_save_fname='22Germany_img_str_list_16740.json'
img_str_save_path=os.path.join(save_dir,img_str_save_fname)
if (not os.path.exists(img_str_save_path)) or (overwrite_flag == 1): 
    with open(img_str_save_path, "w", encoding='utf-8') as f:
        json.dump(img_str_list, f, indent=2)
    print('{} saved!'.format(img_str_save_fname))


# 检查 是否1654类 每类10张； 200类 每类1张
for i in range(1654):
    print(img_str_list[i*10])

aaa=set(img_str_list)
len(aaa)==16740
