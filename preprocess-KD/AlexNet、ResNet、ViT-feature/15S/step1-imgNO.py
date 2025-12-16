import os
import json
import numpy as np


# 0、构建img_str与imgNO的对应表 (     个图片名构成的列表，其位置就是imgNO号)

img_dir = '/data/snxue/visual_embedding的图片/72pic/'
img_str_list=os.listdir(img_dir)


overwrite_flag=1
save_dir = '/data/snxue/eeg_preprocess/15Stanford-img/'
img_str_save_fname='15Stanford_img_str_list_72.json'
img_str_save_path=os.path.join(save_dir,img_str_save_fname)
if (not os.path.exists(img_str_save_path)) or (overwrite_flag == 1): 
    with open(img_str_save_path, "w", encoding='utf-8') as f:
        json.dump(img_str_list, f, indent=2)
    print('{} saved!'.format(img_str_save_fname))


