import os
import json
import numpy as np


img_path = "/share/eeg_datasets/Vision/21_Purdue/CVPR2021-02785/stimuli/"
img_list = os.listdir(img_path)
img_list.sort()

# 0、构建img_str与imgNO的对应表 (40000个图片名构成的列表，其位置就是imgNO号)
overwrite_flag=0
save_dir = '/data/snxue/eeg_preprocess/21Purdue-img/'
img_str_save_fname='21Purdue_img_str_list_40000.json'
img_str_save_path=os.path.join(save_dir,img_str_save_fname)
if (not os.path.exists(img_str_save_path)) or (overwrite_flag == 1): 
    with open(img_str_save_path, "w", encoding='utf-8') as f:
        json.dump(img_list, f, indent=2)
    print('{} saved!'.format(img_str_save_fname))


# 检查 是否40类 每类1000张  ImageNet的命名规律
for i in range(40*2):
    print(img_list[i*500])

