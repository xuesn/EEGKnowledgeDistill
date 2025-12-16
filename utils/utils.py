

# main_dir = '/share/models/snxue/'  # share loss路径
main_dir = '/data/snxue/' # 98 240 data 路径  0311-15S
main_dir = '/data1/snxue/' # 98 data 路径  0313-21P 22G
# main_dir = '/data/xueshuning/'  # 模识楼   1
# main_dir = '/home/nettrix/xueshuning/'  # 模识楼   2
main_dir = '/mnt/data/snxue/' # 43 44 45 路径


from colorama import init, Fore,  Back,  Style
init(autoreset = True)

from model.model_linear import *
from model.model_conv import *
from model.model_ShallowConvNet import *
from model.model_EEGNet import *
from model.model_MetaEEG import *
from model.model_NICE import *
from model.model_ATMS_50 import *
from model.model_SCT import *
from model.model_SCT_pool import *


# dataset_str 不使用
def get_model_encoder(model_str, dataset_str, 
        timepoint_num, electrode_num, 
        class_num, 
        subject_num,
        vis_emb_dim):        
    if model_str == 'linear':
        in_dim = timepoint_num * electrode_num
        out_dim = vis_emb_dim  
        model = ModelLinear(in_dim,out_dim)
        # linear不需要在外面接vis-proj-head
    elif model_str == 'conv':
        model = ModelConv(ch1=16, ch2=32, ch3=64, ch4=64,
                 kernal1=5, kernal2=5, kernal3=5, kernal4=124,
                 pool_stride=2,
                 dropout_p=0.25,
                 activate_type='elu',
                 pool_type='max',
                 electrode_num=electrode_num, )
        # 需要在外面接vis-proj-head
    elif model_str == 'ShallowConvNet':
        model = ShallowConvNet(chans=electrode_num, 
                    samples=timepoint_num)
    elif model_str == 'EEGNet':
        model = EEGNet(chans=electrode_num,  samples=timepoint_num, dropout_rate=0., kernel_length=4, F1=16,
                 F2=64,)
    # elif model_str == 'EEGNetv4':
    #     model = EEGNetv4_Encoder()
    elif model_str == 'MetaEEG':
        model = MetaEEG(num_channels=electrode_num, sequence_length=timepoint_num, num_subjects=subject_num, num_features=64, num_latents=1024, num_blocks=1)
        # num_latents是MLP的参数，这里不用
    elif model_str == 'NICE':
        model = NICE(electrode_num,a_dim = 256) #electrode_num,a_dim = 64这是我自己添加的参数
    elif model_str == 'ATMS_50':
        model = ATMS_50(num_channels=electrode_num, sequence_length=timepoint_num, num_subjects=subject_num, num_features=256, num_latents=1024, num_blocks=1)
        # num_latents是MLP的参数，这里不用
    elif model_str == 'SCT':  # 22A 22G不pool性能好
        model = model_SCT(ch1=16, ch2=32, ch3=64, ch4=64,
                kernal1=5, kernal2=5, kernal3=5, kernal4=electrode_num,
                pool_stride=9999,
                dropout_conv=0.25,
                classify_fc_hid=None, class_num=None,
                activate_type='elu',
                pool_type='mean',
                timepoint_num=timepoint_num,electrode_num=electrode_num,                   
                trans_layers=1, trans_fc_hid=16*16,  trans_head=16,
                act_func_trans='relu', dropout_trans=0.1,
                ch_dim_reduce=8,
                decoder_depth=0, decoder_embed_dim=64, decoder_num_heads=1,
                vis_emb_dim=vis_emb_dim )
    elif model_str == 'SCT_pool':        # 15S 21P pool性能好
        model = model_SCT_pool(ch1=16, ch2=32, ch3=64, ch4=64,
                kernal1=5, kernal2=5, kernal3=5, kernal4=electrode_num,
                pool_stride=2,
                dropout_conv=0.25,
                classify_fc_hid=None, class_num=None,
                activate_type='elu',
                pool_type='mean',
                timepoint_num=timepoint_num,electrode_num=electrode_num,                   
                trans_layers=1, trans_fc_hid=16*16,  trans_head=16,
                act_func_trans='relu', dropout_trans=0.1,
                ch_dim_reduce=8,
                decoder_depth=0, decoder_embed_dim=64, decoder_num_heads=1,
                vis_emb_dim=vis_emb_dim )
    return model


# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
def get_pic_emb(pic_emb_dir, dataset_str, vis_emb_str, vis_emb_type):        
    if dataset_str in ['15Stanford_6class' , '15Stanford_72pic']:
        dataset_dir = '15Stanford'
    elif dataset_str in ['21Purdue']:
        dataset_dir = '21Purdue'
    elif dataset_str in ['22Germany_train','22Germany_test','22Germany_mean']:
        dataset_dir = '22Germany'
    elif dataset_str in ['22Australia_train','22Australia_test','22Australia_mean']:
        dataset_dir = '22Australia'
    else:
        assert 'Unrecognized dataset_str'
    #
    vis_emb_str_list = ['alex', 'resnet18', 'resnet34',
                  'resnet50', 'resnet101', 'vit_b_16']
    feat_dim_dict={}
    feat_dim_dict['alex']=9216
    feat_dim_dict['resnet18']=512
    feat_dim_dict['resnet34']=512
    feat_dim_dict['resnet50']=2048
    feat_dim_dict['resnet101']=2048
    feat_dim_dict['vit_b_16']=768
    #
    if vis_emb_type == 'image':
        pic_emb_fname = vis_emb_str+'_dim'+str(feat_dim_dict[vis_emb_str])+'.npy'
    elif vis_emb_type == 'label':
        pic_emb_fname = vis_emb_str+'_dim'+str(feat_dim_dict[vis_emb_str])+'_label.npy'
    pic_emb_path = os.path.join(pic_emb_dir,dataset_dir,pic_emb_fname)
    pic_emb = np.load(pic_emb_path)
    
    return pic_emb,  feat_dim_dict[vis_emb_str]

# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
def test_model_output_shape(model,rank,in_shape):
    # 主要用于测试fc的输入维度
    test_bs =2 # 取batchsize为2
    temp = in_shape.copy()
    temp.insert(0, test_bs)
    # in_shape.insert(index=0, obj=test_bs) #TypeError: list.insert() takes no keyword arguments
    aaa=torch.rand(temp).to(rank)
    output = model(aaa)
    output = list(output)
    bbb = output[0]  # 默认模型输出的第一个为embedding
    out_shape = bbb.shape[1:] #[1:]为了去除bs维度
    # 得到fc的输入维度（除bs维度外，相乘）
    fc_in_dim = 1
    for i in out_shape:
        fc_in_dim*=i
    return out_shape,fc_in_dim

# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
import time
#将10位时间戳转换为时间字符串，默认为2017-10-01 13:37:04格式
def timestamp_to_date(time_stamp, format_string="%Y-%m-%d-%H-%M-%S"):
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date

# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
from dataset.dataset_15Stanford import *
from dataset.dataset_21Purdue import *
from dataset.dataset_22Germany_mean_nopreload import *
# from dataset.dataset_22Australia_mean_nopreload import *

def get_all_sub_list(dataset_str):
    sub_list_list = None    
    if dataset_str == '15Stanford_6class':
        sub_list_list = [['S1'],['S2'],['S3'],['S4'],['S5'],
                         ['S6'],['S7'],['S8'],['S9'],['S10']]
    elif dataset_str == '15Stanford_72pic':
        sub_list_list = [['S1'],['S2'],['S3'],['S4'],['S5'],
                         ['S6'],['S7'],['S8'],['S9'],['S10']]    
    elif dataset_str == '21Purdue':    
        # sub_list_list = [['exp00','exp01','exp02','exp03','exp04','exp05','exp06','exp07','exp08','exp09',
        #                   'exp10','exp11','exp12','exp13','exp14','exp15','exp16','exp17','exp18','exp19',
        #                   'exp20','exp21','exp22','exp23','exp24','exp25','exp26','exp27','exp28','exp29',
        #                   'exp30','exp31','exp32','exp33','exp34','exp35','exp36','exp37','exp38','exp39',
        #                   'exp40','exp41','exp42','exp43','exp44','exp45','exp46','exp47','exp48','exp49',
        #                   'exp50','exp51','exp52','exp53','exp54','exp55','exp56','exp57','exp58','exp59',
        #                   'exp60','exp61','exp62','exp63','exp64','exp65','exp66','exp67','exp68','exp69',
        #                   'exp70','exp71','exp72','exp73','exp74','exp75','exp76','exp77','exp78','exp79',
        #                   'exp80','exp81','exp82','exp83','exp84','exp85','exp86','exp87','exp88','exp89',
        #                   'exp90','exp91','exp92','exp93','exp94','exp95','exp96','exp97','exp98','exp99',]]
        sub_list_list = [['']]
    elif dataset_str in ['22Germany_train','22Germany_test','22Germany_mean']:
        sub_list_list=[['sub-01'],['sub-02'],['sub-03'],['sub-04'],['sub-05'],
                ['sub-06'],['sub-07'],['sub-08'],['sub-09'],['sub-10']]

    elif dataset_str in ['22Australia_train','22Australia_test','22Australia_mean']:
        sub_list_list=[            ['sub-02'],['sub-03'],['sub-04'],['sub-05'],           ['sub-07'],['sub-08'],['sub-09'],['sub-10'],
                        ['sub-11'],['sub-12'],['sub-13'],['sub-14'],['sub-15'],['sub-16'],['sub-17'],['sub-18'],['sub-19'],['sub-20'],
                        ['sub-21'],['sub-22'],['sub-23'],['sub-24'],['sub-25'],['sub-26'],['sub-27'],['sub-28'],['sub-29'],['sub-30'],
                        ['sub-31'],['sub-32'],['sub-33'],['sub-34'],['sub-35'],['sub-36'],['sub-37'],['sub-38'],['sub-39'],['sub-40'],
                        ['sub-41'],['sub-42'],['sub-43'],['sub-44'],['sub-45'],['sub-46'],['sub-47'],['sub-48']]
    return sub_list_list

def choose_dataset(dataset_str):
    sub_list_list = get_all_sub_list(dataset_str)
    if dataset_str == '15Stanford_6class':
        func_dataset = dataset_15Stanford_6class
    elif dataset_str == '15Stanford_72pic':
        func_dataset = dataset_15Stanford_72pic
    elif dataset_str == '21Purdue':
        func_dataset = dataset_21Purdue
    elif dataset_str in ['22Germany_train','22Germany_test','22Germany_mean']:
        func_dataset = dataset_22Germany_mean_train
    elif dataset_str in ['22Australia_train','22Australia_test','22Australia_mean']:
        func_dataset = dataset_22Australia_mean_train
    return func_dataset, sub_list_list
    
# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
import torch
import random
import numpy as np
# 随机种子
def seed_everything(seed):
    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)
    
    torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。

# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
from accelerate import Accelerator, DistributedType
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import logging
#多卡 分布式
def initial_dist(rank, world_size,cudaNO_list):
    # 分布式 初始化
    # 下面这一句有时候卡死
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size, )
    # dist.init_process_group("nccl", init_method='tcp://localhost:23456', rank=rank, world_size=world_size, )
    if torch.distributed.get_rank() == 0:  #可以选任意的rank保存。
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # 显卡
    logging.info('CUDA:{}'.format(cudaNO_list))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    accelerator_device = accelerator.device
    logging.info('accelerator_device:{}'.format(accelerator_device))
    distributed_type = accelerator.distributed_type
    logging.info('distributed_type:{}'.format(distributed_type))
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    return device,accelerator,scaler

# - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ * - ~ 
import os
#续训模型
def load_checkpoint(
    target_ckpt_list,
    save_path_checkpoint,checkpoint_fname_rear,
    model,optimizer
    ):
    if len(target_ckpt_list) == 1:
        ckpt_fname = target_ckpt_list[0]
        checkpoint_path = os.path.join(
            save_path_checkpoint, ckpt_fname)
        chechpoint = torch.load(checkpoint_path,map_location='cpu')
        model.load_state_dict(chechpoint['model_state_dict'])
        optimizer.load_state_dict(chechpoint['optimizer_state_dict'])
        val_acc_or_loss = chechpoint['val_acc_or_loss']
        epoch = chechpoint['epoch']
        logging.info('已装载：{} loaded!'.format(checkpoint_path))
    elif len(target_ckpt_list) == 0:
        assert False,'No matched checkpoint_path!'
    else:
        assert False,'More than 1 matched checkpoint_path!'
    return model,optimizer,val_acc_or_loss,epoch

def load_checkpoint_from_assigned_epoch(
    start_epoch,
    save_path_checkpoint,checkpoint_fname_rear,
    model,optimizer
    ):
    val_acc_or_loss=None

    checkpoint_list = os.listdir(save_path_checkpoint)
    target_ckpt_list1 = [
        fn for fn in checkpoint_list if fn.endswith(checkpoint_fname_rear)]
    target_ckpt_list2 = [
        fn for fn in target_ckpt_list1 if fn.startswith("Current-val_")]
    epoch_str = '_epoch-'+str(start_epoch)
    target_ckpt_list3 = [
        fn for fn in target_ckpt_list2 if epoch_str in fn]
        
    model,optimizer,val_acc_or_loss,epoch=load_checkpoint(
                                        target_ckpt_list3,
                                        save_path_checkpoint,checkpoint_fname_rear,
                                        model,optimizer
                                        )
    return model,optimizer,val_acc_or_loss,epoch

def load_checkpoint_from_latest_model(        
    save_path_checkpoint,checkpoint_fname_rear,
    model,optimizer):
    val_acc_or_loss = None
    start_epoch=0
    if os.path.exists(save_path_checkpoint):
        checkpoint_list = os.listdir(save_path_checkpoint)
        target_ckpt_list1 = [
            fn for fn in checkpoint_list if fn.endswith(checkpoint_fname_rear)]
        target_ckpt_list2 = [
            fn for fn in target_ckpt_list1 if fn.startswith("Current-val_")]
        epoch_int_list=  [int(x.split('_epoch-')[-1].split('_')[0]) for x in target_ckpt_list2]#有时候中断时没来得及删掉前一个epoch，会有2个checkpoint
        if len(epoch_int_list)!=0:
            max_epoch=max(epoch_int_list)
            start_epoch=max_epoch
            epoch_str = '_epoch-'+str(max_epoch)
            target_ckpt_list3 = [
                fn for fn in target_ckpt_list2 if epoch_str in fn]
                
            model,optimizer,val_acc_or_loss,start_epoch=load_checkpoint(
                                                target_ckpt_list3,
                                                save_path_checkpoint,checkpoint_fname_rear,
                                                model,optimizer
                                                )
    return model,optimizer,val_acc_or_loss,start_epoch

