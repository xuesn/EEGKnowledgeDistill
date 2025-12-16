import os
import json
import numpy as np


# 4w张图片的编号顺序
img_path = "/share/eeg_datasets/Vision/21_Purdue/CVPR2021-02785/stimuli/"
img_list = os.listdir(img_path)
img_list.sort()

# 不分run保存
img_str_dir = "/share/eeg_datasets/Vision/21_Purdue/CVPR2021-02785/design/"
img_str_list = os.listdir(img_str_dir)
img_str_list = [fn for fn in img_str_list if fn.endswith('.txt')]
img_str_list = [fn for fn in img_str_list if fn.startswith('run-')]
img_str_list.sort()
#
imgNO_dataset = []
for img_str_fname in img_str_list:
    img_str_fpath = os.path.join(img_str_dir,img_str_fname)
    with open(img_str_fpath, "r", encoding='utf-8') as f:
        img_str_this_run = f.read()
    img_str_this_run = img_str_this_run.split('\n')
    len(img_str_this_run) == 401
    img_str_this_run[-1]
    img_str_this_run = img_str_this_run[:-1]
    len(img_str_this_run) == 400
    img_str_this_run[-1]
    # break

    imgNO_this_run = [ img_list.index(img_str)    for img_str in img_str_this_run]
    imgNO_dataset += imgNO_this_run
    print(img_str_fname,'added!')

imgNO_dataset_arr = np.array(imgNO_dataset,np.int32).reshape([-1,1])
# save
save_dir = '/data/snxue/eeg_preprocess/21Purdue-img/'
imgNO_save_fname='21Purdue_imgNO.npy'
imgNO_save_path=os.path.join(save_dir,imgNO_save_fname)
overwrite_flag = 0
if (not os.path.exists(imgNO_save_path)) or (overwrite_flag == 1): 
    np.save(imgNO_save_path,imgNO_dataset_arr)
    print('{} saved!'.format(imgNO_save_fname))


np.min(imgNO_dataset_arr) == 0
np.max(imgNO_dataset_arr) == 39999
for i in range(40000):
    if i not in imgNO_dataset_arr:
        print(i,'not exist!')


