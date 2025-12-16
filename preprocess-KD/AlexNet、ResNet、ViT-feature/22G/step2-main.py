




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


# 3、读取图片
preprocessed_img_save_path='/data/snxue/eeg_preprocess/22Germany-img/'+'22Germany_normed_img_arr.npy'

img_set=np.load(preprocessed_img_save_path)
print('img_set',' loaded!')
'''

# 设置放缩后图像大小
img_height = 224
img_width = 224
RGBnum = 3
img_set = np.zeros([img_num, img_width, img_height, RGBnum])
# 均值、标准差
# img_mean= [137.93283222269082, 126.14478701041628, 111.6941045191178]
# img_std= [68.59019955675515, 66.13754000357918, 70.16372693915085]
#
img_mean= [0.5409130675399693, 0.49468543925653197, 0.4380160961534088]
img_std= [0.2689811747323737, 0.25936290197481887, 0.2751518703496101]

# iva23的图片
# img_mean= [0.47698926,0.45736344, 0.41217181]
# img_std= [0.26331587 ,0.25778438 ,0.27545269]

# 路径
for imgNO, img_path in enumerate(img_path_list):
    img = Image.open(img_path)
    img_arr = np.array(img.resize([img_width, img_height]))
    # 灰度图像转RGB就是把通道复制3遍
    if len(img_arr.shape) == 2:
        img_arr = np.expand_dims(img_arr,2).repeat(3,axis=2)
    img_set[imgNO, :, :, :] = (img_arr/255-img_mean)/img_std
    print('img',imgNO,' loaded!')

# # 计算mean std
# img_mean_r = np.mean(img_set[:,:,:,0])
# img_mean_g = np.mean(img_set[:,:,:,1])
# img_mean_b = np.mean(img_set[:,:,:,2])
# img_mean = [img_mean_r,img_mean_g,img_mean_b]

# img_std_r = np.std(img_set[:,:,:,0])
# img_std_g = np.std(img_set[:,:,:,1])
# img_std_b = np.std(img_set[:,:,:,2])
# img_std = [img_std_r,img_std_g,img_std_b]


np.save(preprocessed_img_save_path,img_set.astype(np.float32))
print('!!!'+preprocessed_img_save_path+' saved!!!')
#6G~5742.1875MB = 10000张 * 3*224*224（3*50176）  /1024/1024
#10G~9843.120MB = 16740张 * 3*224*224（3*50176）  /1024/1024
'''

# 4、提取特征

# model装载模型
finished_num = 0
model_str_list = ['alex', 'resnet18', 'resnet34',
                  'resnet50', 'resnet101', 'vit_b_16']
for model_str in model_str_list[finished_num:]:
    start_time = time.time()
    # 选择模型----------------------------------------------------------------------------------------------------
    # 只提特征，去分类头
    if model_str == 'alex':
        model = models.alexnet(pretrained=True)
        img_feat_dim = model.classifier[1].in_features  # 9216
        model.classifier = torch.nn.Sequential()
    # elif model_str == 'resnet18' | model_str == 'resnet34'| model_str == 'resnet50'| model_str == 'resnet101' :
    elif model_str == 'resnet18':
        model = models.resnet18(pretrained=True)
        img_feat_dim = model.fc.in_features  # 512
        model.fc = torch.nn.Sequential()
    elif model_str == 'resnet34':
        model = models.resnet34(pretrained=True)
        img_feat_dim = model.fc.in_features  # 512
        model.fc = torch.nn.Sequential()
    elif model_str == 'resnet50':
        model = models.resnet50(pretrained=True)
        img_feat_dim = model.fc.in_features  # 2048
        model.fc = torch.nn.Sequential()
    elif model_str == 'resnet101':
        model = models.resnet101(pretrained=True)
        img_feat_dim = model.fc.in_features  # 2048
        model.fc = torch.nn.Sequential()
    elif model_str == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
        img_feat_dim = model.heads.head.in_features  # 768
        model.heads = torch.nn.Sequential()
    else:
        assert 1 != 1, '其他模型没做哦~~~'
    print(model_str,'loaded!',time.time()-start_time)


    # 模型提取特征----------------------------------------------------------------------------------------------------
    # 提取原始脑电数据的特征
    # model_enc_proj = new_model  # 预装载好权重的enc模型  new_model太麻烦 不如直接把分类头改为sequence或dropout
    model_enc_proj = model  # 预装载好权重的enc模型
    model_enc_proj.to(device)
    sample_feature = np.zeros([img_num, img_feat_dim])
    num_workers = 1
    tensor_img_set = torch.tensor(img_set, dtype=torch.float32)
    if (model_str == 'alex') | (model_str == 'vit_b_16') | (model_str == 'resnet18') | (model_str == 'resnet34') | (model_str == 'resnet50') | (model_str == 'resnet101'):
        # vit/resnet的输入要为 n, c, h, w = x.shape
        tensor_img_set = tensor_img_set.permute(0, 3, 2, 1)

    # else:

    # dataloader
    encode_batch_size = 512
    start_time = time.time()
    encode_set = torch.utils.data.TensorDataset(
        tensor_img_set, )    
    encode_loader = torch.utils.data.DataLoader(encode_set, encode_batch_size,
                                                shuffle=False, num_workers=2)
    # shuffle=False可一定要注意啊，这里如果打乱了，label就全乱了
    print('dataloader constructed!',time.time()-start_time)




    # 开始提取----------------------------------------------------------------------------------------------------
    model_enc_proj.eval()
    with torch.no_grad():  # 注意一定要加 不然会保存梯度 占大量GPU
        for step, (batch_x,) in enumerate(encode_loader):  # 要写成(batch_x,)，不能(batch_x)哦
            # break
            batch_x = batch_x.to(device)  # bs=50 大概占800
            out = model_enc_proj(batch_x)  # bs=50 大概又占800
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





# # 5、训练linear
# new_feat_save_path = '/data/snxue/visual_embedding/22Germany/prob/'

# feat_dim_dict={}
# feat_dim_dict['alex']=9216
# feat_dim_dict['resnet18']=512
# feat_dim_dict['resnet34']=512
# feat_dim_dict['resnet50']=2048
# feat_dim_dict['resnet101']=2048
# feat_dim_dict['vit_b_16']=768

# learn_rate = 1e-3
# start_epoch = 0
# num_epoches = 100
# criterion_ce = nn.CrossEntropyLoss()


# # 随机种子
# seed = 557
# def seed_everything(seed):
#     #  下面两个常规设置了，用来np和random的话要设置 
#     np.random.seed(seed) 
#     random.seed(seed)
    
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
#     torch.manual_seed(seed)
#     '''在这里使用下列控制seed的方法会有报错
#         RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


#     os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    
#     torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
#     torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
#     torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
#     torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。
#     '''


# # ----------------------------------------------------------------------------------------------------
# #读取 原特征 和 对应图片类别
# vis_feat_path='/data/snxue/visual_embedding/22Germany/'

# model_str_list = ['alex', 'resnet18', 'resnet34',
#                   'resnet50', 'resnet101', 'vit_b_16']
# for model_str in model_str_list:
#     ori_feat_dim=feat_dim_dict[model_str]
#     npy_fname=model_str+'_dim'+str(ori_feat_dim)+'.npy'
#     sample_feature=np.load(os.path.join(vis_feat_path,npy_fname))
#     print(npy_fname,'loaded!')
    
#     sample_num= sample_feature.shape[0]

#     encode_batch_size = sample_num
#     trainset = torch.utils.data.TensorDataset(torch.tensor(sample_feature), torch.tensor(label_onehot))
#     seed_everything(seed)
#     trainloader = torch.utils.data.DataLoader(trainset, encode_batch_size,shuffle=False, )

#     # linear分类网络
#     new_feat_dim = class_num

#     fc_in_dim = ori_feat_dim
#     # classify_fc_hid = new_feat_dim
#     fc_out_dim = class_num

#     seed_everything(seed)
#     model_linear=nn.Sequential(
#         nn.Linear(fc_in_dim, fc_out_dim),
#         nn.Softmax(dim=-1)    ).to(device)
#     # 优化器
#     optimizer_linear = optim.Adam(model_linear.parameters(), lr=learn_rate,)  
#     # ----------------------------------------------------------------------------------------------------
#     # 训练
#     for epoch in range(start_epoch, num_epoches):
#         new_feat_arr = np.zeros([sample_num,new_feat_dim])

#         # break
#         train_right = 0
#         train_loss = 0
#         train_num = trainloader.dataset.tensors[0].shape[0]
#         train_batch_num = len(trainloader)
#         # 每一步loader释放一小批数据用来学习
#         for step, (batch_x, batch_y) in enumerate(trainloader):
#             # break
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             batch_x = torch.tensor(batch_x, dtype=torch.float32)
#             batch_y = torch.tensor(batch_y, dtype=torch.float32)
#             # out, emb = model_linear(batch_x)
#             out = model_linear(batch_x)
#             loss_ce = criterion_ce(out, batch_y.float())
#             # 每个batch更新一次参数
#             optimizer_linear.zero_grad()
#             loss_ce.backward()
#             optimizer_linear.step()
            
#             # for print
#             train_loss = train_loss+loss_ce.item()
#             # 返回最大值的位置，注意是从0开始的
#             pred = np.argmax(
#                 out.cpu().detach().numpy(), axis=1)
#             # 返回最大值的位置，注意是从0开始的
#             label = np.argmax(
#                 batch_y.cpu().detach().numpy(), axis=1)
#             train_right = train_right + \
#                 float(sum(label == pred))

#             #填入数组
#             emb = out
#             index_st = step * encode_batch_size
#             index_end = max(     (step+1) * encode_batch_size,   sample_num)
#             new_feat_arr[index_st:index_end,:] = emb.cpu().detach().numpy()
#     #############################################
#         # 打印训练整数epoch后的loss和acc
#         train_loss = train_loss/train_batch_num
#         train_acc = train_right/train_num
#         print("epoch:", epoch,
#             " train_loss:", train_loss,
#             " train_acc:", train_acc,)
#     #############################################
#         # 保存每个epoch的probability，作为软标签
    
#         if epoch % 10 ==9:
#             # 保存特征/probability----------------------------------------------------------------------------------------------------
#             new_feat_fname = model_str+'_dimChangeTo_'+str(new_feat_dim) + \
#                 '_lr-'+str(learn_rate) + \
#                 '_bs-' + str(encode_batch_size) + \
#                 '_epoch-'+str(epoch) + \
#                 '_seed-'+str(seed) + \
#                 '_acc-'+str(train_acc) + \
#                 '.npy'
#             np.save(os.path.join(new_feat_save_path,new_feat_fname),new_feat_arr)
#             print('!!!', new_feat_fname, ' saved!!!')












# alex 特征维度为 9216 。而sample_feature中有 103932972 个零值。样本数为 16740
# resnet18 特征维度为 512 。而sample_feature中有 116934 个零值。样本数为 16740
# resnet34 特征维度为 512 。而sample_feature中有 82763 个零值。样本数为 16740
# resnet50 特征维度为 2048 。而sample_feature中有 171690 个零值。样本数为 16740
# resnet101 特征维度为 2048 。而sample_feature中有 196267 个零值。样本数为 16740

# vit_b_16 特征维度为 768 。而sample_feature中有 0 个零值。样本数为 16740
# 100MB





