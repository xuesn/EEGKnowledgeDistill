#添加各种loss
# 1、ce分类
# 2、mask：AE
# 3、contrast：a、有监督对比 b、triplet(原，mask或其他增强，负)
# 4、视觉监督




target_fold_list = [0,1,2,3,4,5,6,7,8,9]
target_fold_list = [target_fold_list[替换用]]

import os
cudaNO_list = [      替换用 % 8       ]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cudaNO_list))  # 一般在程序开头设置
os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '123'+str(cudaNO_list[0])+str(cudaNO_list[0])
os.environ['MASTER_PORT'] = '123'+str(cudaNO_list[0])+str(cudaNO_list[0]+1)
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# 常用库 + torch
import time
import logging
from colorama import init, Fore,  Back,  Style
init(autoreset = True)
from sklearn.model_selection import StratifiedKFold,KFold
import torch
from torch import optim
from torch.optim import lr_scheduler
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
#分布式 混合精度
import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
#my-code #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from parameter.parameter_dataset import *
from parameter.parameter_train import *
from parameter.parameter_save import *
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# from dataset.dataset_22Germany_mean_preload import dataset_22Germany_mean
from dataset.dataset_preload import My_Dataset_preload_split, My_Dataset_preload_merge
from dataset.dataset_nopreload import My_Dataset_nopreload_split, My_Dataset_nopreload_merge
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from model.fc_model import ProjNet_FC
from model.model_linear import ModelLinear
from model.reweight_model import ModelLinearReweight
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from train.train_supervised_contrast_visual import *
# from train.train_unsupervised_contrast_visual import *
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from save.utils_loss_acc_dict import *
from save.utils_save import *
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from utils.utils import *





from torch.cuda.amp import GradScaler
from accelerate import Accelerator, DistributedType

scaler = GradScaler()
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator()
accelerator_device = accelerator.device
print('accelerator_device:{}'.format(accelerator_device))
distributed_type = accelerator.distributed_type
print('distributed_type:{}'.format(distributed_type))


# 
world_size =  len(cudaNO_list)
rank=0


# 一、parameter
# dataset
clamp_thres, norm_type, mask_num, mask_len, sub_list = parameter_dataset()

# train
seed, start_epoch, end_epoch, load_latest_model, csv_overwrite_flag, train_batch_size, val_batch_size = parameter_train()
# regularize
dropout_trans, dropout_conv, dropout_fc, weight_decay = parameter_regularize()
# optimizer
beta_1, beta_2, eps, learn_rate, gamma = parameter_optimizer()




# ~ ~ ~ ~ ~ ~
# 选择数据集
# 15Stanford_6class
# 15Stanford_72pic
# 21Purdue
# 22Australia_mean
# 22Germany_mean
dataset_str = '21Purdue'
func_dataset,sub_list_list = choose_dataset(dataset_str)

single_multi_cross = 'singleSub'


# ~ ~ ~ ~ ~ ~

# loss
(dlt_ae,loss_ae_weight,
temp_eeg,loss_spc_weight,margin_eeg,loss_triplet_weight,
delta_vis, temperature_vis, loss_vis_huber_weight, loss_vis_infonce_weight, loss_vis_kl_weight, vis_emb_str, vis_emb_type, vis_reweight_type, vis_layer_str,
pic_emb, vis_emb_dim, vis_emb_num,
flag_layerwise_or_reweight,
criterion_ce,criterion_ae,criterion_spc,criterion_triplet, 
criterion_huber, criterion_infonce, criterion_kl) = parameter_loss(dataset_str)




# 15S6  15S72  21P
lr_list = [ 1e-3 ]
bs_list = [ 32 ]
    # train_batch_size = 32  # 15S 21P
    # train_batch_size = 125 # 22G

wd_list = [ 0 ]
dp_fc_list = [ 0 ]
dp_conv_list = [ 0.25 ]

loss_spc_weight_list = [ 0,1 ]
loss_spc_weight_list = [ 0 ]

loss_triplet_weight_list = [ 0,1]
loss_triplet_weight_list = [ 1]
margin_eeg_list = [ 1 ]
    # margin_eeg = 1  # 15S  21P
    # margin_eeg = 0  # 22G 

loss_vis_huber_weight_list = [  50 ]
    # loss_vis_huber_weight = 50   # 15S 21P
    # loss_vis_huber_weight = 100  # 22G

delta_vis_list = [ 0.5 ]
    # delta_vis = 0.5   # 15S 21P
    # delta_vis = 0.4   # 22G


loss_vis_infonce_weight_list = [ 0 ]

loss_vis_kl_weight_list = [ 0 ]
# 逐一配对版

parameter_list_whole = [ (a,b,c,d,e,f,g,
                            h,i,j,k,l,)    
                            for a in   lr_list 
                            for b in   bs_list 
                            for c in   wd_list
                            for d in   dp_fc_list 
                            for e in   dp_conv_list 
                            for f in   loss_spc_weight_list 
                            for g in   loss_triplet_weight_list 
                            for h in   margin_eeg_list 
                            for i in   loss_vis_huber_weight_list 
                            for j in   delta_vis_list 
                            for k in   loss_vis_infonce_weight_list 
                            for l in   loss_vis_kl_weight_list   ] 
# print(parameter_list_whole)
# parameter_list_whole = []
parameter_list_this_cuda = parameter_list_whole[:]
for ( learn_rate, train_batch_size , 
      weight_decay , dropout_fc , dropout_conv , 
      loss_spc_weight , loss_triplet_weight , margin_eeg , 
      loss_vis_huber_weight  , delta_vis , loss_vis_infonce_weight , loss_vis_kl_weight ) in parameter_list_this_cuda:
    # loss函数需要在内部修改
    criterion_triplet = BatchAllTripletLoss(margin_eeg)
    criterion_huber = nn.HuberLoss(reduction='mean', delta=delta_vis)


    # ~ 根据前面的参数，设定保存的parameter_str_short ~ ~ ~ ~ ~
    # save
    save_code_dir, save_model_dir, save_dataset_dir, parameter_str_short, save_dir_ckpt, save_dir_csv, save_dir_csv_batch = parameter_save(learn_rate,train_batch_size,world_size,seed,mask_num,mask_len,dlt_ae,loss_ae_weight,temp_eeg,loss_spc_weight,margin_eeg,loss_triplet_weight,
    delta_vis, temperature_vis, loss_vis_huber_weight, loss_vis_infonce_weight, loss_vis_kl_weight, vis_emb_str, vis_emb_type, vis_reweight_type, vis_layer_str,
    flag_layerwise_or_reweight,
    dataset_str, single_multi_cross,)
    # dropout_fc,dropout_conv)


    # 归一化
    norm_per_sample = False
    norm_per_electrode = False
    norm_per_2sample_electrode = False
    if norm_type=='norm_per_sample':
        norm_per_sample = True
        norm_type_short= 'nps'
    elif norm_type=='norm_per_electrode':
        norm_per_electrode = True
        norm_type_short= 'npe'
    elif norm_type=='norm_per_2sample_electrode':
        norm_per_2sample_electrode = True
        norm_type_short= 'np2se'
    elif norm_type=='none':
        norm_per_2sample_electrode = True
        norm_type_short= 'none'


    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # 二、data    选择模型   

    # 选择模型
    model_str_list=['linear', 'conv', 'ShallowConvNet', 'EEGNet', 'MetaEEG', 'NICE', 'ATMS_50', 'SCT','SCT_pool', ]
    model_str_list=['SCT_pool', ]
    # 22A 22G不pool性能好   # 15S 21P pool性能好


    from itertools import product
    result = product(sub_list_list, model_str_list, )
    for sub_list, model_str in result:
        print('sub_list:',sub_list, '\tmodel_str:', model_str)

        sub_num = len(sub_list)
        save_dir_ckpt_this_sub = save_dir_ckpt.replace(save_dataset_dir,save_dataset_dir+'_subNum'+str(sub_num)+'_'+sub_list[0])
        save_dir_csv_this_sub = save_dir_csv.replace(save_dataset_dir,save_dataset_dir+'_subNum'+str(sub_num)+'_'+sub_list[0])
        save_dir_csv_batch_this_sub = save_dir_csv_batch.replace(save_dataset_dir,save_dataset_dir+'_subNum'+str(sub_num)+'_'+sub_list[0])

        # 以train-set作为whole_set，切分出验证集，做跨模态对比学习
        start_time = time.time()  
        whole_set = func_dataset(sub_list,clamp_thres,
            norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,)
        print(' load time:', time.time()-start_time)
        sample_num=len(whole_set)
        print(' 样本数:', sample_num)    
        
        #样本维度
        eeg,label_onehot,imgNO = whole_set.__getitem__(1)  # 测试，ps：python并无私有
        timepoint_num, electrode_num = eeg.shape
        class_num = label_onehot.shape[0]
        print('timepoint_num:{} electrode_num:{} class_num:{}'.format(timepoint_num, electrode_num, class_num))
            

        #split train/val
        n_splits=10
        split_proportion_list = [1]*n_splits  # 之后就可以每一折交叉验证
        print('随机种子:{}'.format(  seed))
        seed_everything(seed)
        sampleNO_split_list = whole_set.split_dataset(split_proportion_list,seed)

        split_dataset_list = []
        for sampleNO_split in sampleNO_split_list:
            if '15Stanford' in dataset_str:
                dataset_split = My_Dataset_preload_split(
                                whole_set, sampleNO_split)
            else:
                dataset_split = My_Dataset_nopreload_split(
                                whole_set, sampleNO_split)
            split_dataset_list.append(dataset_split)
        #
        for foldNO in range(n_splits):


            # 跳过非指定fold
            if not foldNO in target_fold_list:
                continue
            
            fold_str = '_foldNO-'+str(foldNO).zfill(2) 
            # 文件名(loss)，ckpt的要记录epoch和loss/acc
            time_str = timestamp_to_date(time_stamp=time.time(), format_string = "%Y-%m-%d-%H-%M-%S")
            csv_fname = fold_str+'_'+ time_str + '.csv'
            checkpoint_fname_rear =  fold_str+'_'+ '_checkpoint.pt'


            #重新将start_epoch置零
            start_epoch = 0
            #
            trainset_list=[]
            valset_list=[]
            for i in range(n_splits):
                if i==foldNO:
                    continue
                trainset_list.append(split_dataset_list[i])
            valset_list.append(split_dataset_list[foldNO])
            if '15Stanford' in dataset_str:
                trainset = My_Dataset_preload_merge(trainset_list)
                valset = My_Dataset_preload_merge(valset_list)  
            else:
                trainset = My_Dataset_nopreload_merge(trainset_list)
                valset = My_Dataset_nopreload_merge(valset_list)  
            #
            datasets={ 'train':    trainset,
                        'val':       valset, } 
            dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
            print('whole_size:{} train_size:{} val_size:{}'.format(
                len(whole_set), len(trainset), len(valset)))
            #  
            # val_batch_size=int(train_batch_size*2)#测试时无梯度，可以大一点
            val_batch_size=99999 # 15S  1078
            val_batch_size=1334  # 21P  26324
            val_batch_size=1000  # 21P  19916
            # val_batch_size=1654  # 22G  22580
            print('train_batch_size:{} val_batch_size:{} '.format(
                train_batch_size, val_batch_size))
            # num_workers = 8,
            # pin_memory = True,
            # prefetch_factor = 4,
            trainloader = torch.utils.data.DataLoader(trainset, train_batch_size, shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, val_batch_size,shuffle=False)
            dataloaders = {'train':    trainloader,
                            'val':      valloader, }    



            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            # 三、model   
            print('开始构建模型训练') 
            # model 
            pool_type = 'mean'  # 好很多
            seed_everything(seed)
            subject_num = 1 # MetaEEG用，但是我这里没有区分被试
            model = get_model_encoder(model_str, dataset_str, 
                timepoint_num, electrode_num, 
                class_num, 
                subject_num,
                vis_emb_dim)

            print('Start -- model.to(rank)')
            start_time = time.time()
            model = model.to(rank)
            print('model.to(rank) cost:{}'.format(time.time()-start_time))
            # 优化器
            optimizer = optim.Adam(model.parameters(), lr=learn_rate,
                                betas=(beta_1, beta_2), eps=eps, weight_decay=weight_decay)
            opt_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma) 

            # # ##############
            # #加视觉特征映射头
            in_shape = [timepoint_num, electrode_num ]
            out_shape,proj_in_dim = test_model_output_shape(model,rank,in_shape)
            proj_vis_head_list = []
            proj_optimizer_list = []
            proj_scheduler_list = []
            #
            # 第一层
            hid_dim_list = []
            proj_vis_head_list.append( ProjNet_FC(
                proj_in_dim,vis_emb_dim,
                hid_dim_list,
                activate_func='relu',
                whether_last_layer_act=False,
                last_layer_act_func='relu',
                dropout_fc=0.).to(rank) )
            # 优化器
            proj_optimizer_list.append( optim.Adam(proj_vis_head_list[-1].parameters(), lr=learn_rate,
                                betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay) )
            #! 余弦退火热重启
            proj_scheduler_list.append(  lr_scheduler.ExponentialLR(proj_optimizer_list[-1], gamma) )


            # 第二层到第十层
            if flag_layerwise_or_reweight == 'layerwise':
                proj_layer_num = vis_emb_num
            elif flag_layerwise_or_reweight == 'reweight':
                proj_layer_num = 1
            for proj_layerNO in range(1,proj_layer_num):
                proj_vis_head_list.append( ProjNet_FC(
                    vis_emb_dim,vis_emb_dim,
                    hid_dim_list,
                    activate_func='relu',
                    whether_last_layer_act=False,
                    last_layer_act_func='relu',
                    dropout_fc=0.).to(rank) )
                # 优化器
                proj_optimizer_list.append( optim.Adam(proj_vis_head_list[-1].parameters(), lr=learn_rate,
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay) )
                #! 余弦退火热重启
                proj_scheduler_list.append(  lr_scheduler.ExponentialLR(proj_optimizer_list[-1], gamma) )

            # # ##############
            #加分类头
            fc_in_dim = vis_emb_dim
            fc_out_dim = class_num
            hid_dim_list = [128]
            fc_model=  ProjNet_FC(
                fc_in_dim,fc_out_dim,
                hid_dim_list,
                activate_func='relu',
                whether_last_layer_act=True,
                last_layer_act_func='softmax',
                dropout_fc=0.)
            fc_model = fc_model.to(rank)
            fc_optimizer = optim.Adam(fc_model.parameters(), lr=learn_rate,
                                betas=(beta_1, beta_2), eps=eps, weight_decay=weight_decay)
            fc_scheduler = lr_scheduler.ExponentialLR(fc_optimizer, gamma)
            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
            model_list = [model]+proj_vis_head_list+[fc_model]
            optimizer_list = [optimizer]+proj_optimizer_list+[fc_optimizer]
            opt_scheduler_list =[opt_scheduler]+proj_scheduler_list+[fc_scheduler]



            # #加视觉特征加权头
            # 逐层学习（包括只有普通的最后一层学习）时不用，置为None
            if flag_layerwise_or_reweight == 'layerwise':
                vis_reweight_model,vis_reweight_optimizer,vis_reweight_scheduler=None,None,None
            elif flag_layerwise_or_reweight == 'reweight':
                # token加权或layer加权
                in_shape = [ vis_emb_num, vis_emb_dim ]
                vis_reweight_model = ModelLinearReweight(vis_emb_num,1).to(rank)
                vis_reweight_optimizer = optim.Adam(vis_reweight_model.parameters(), lr=learn_rate,
                                    betas=(beta_1, beta_2), eps=eps, weight_decay=weight_decay)
                vis_reweight_scheduler = lr_scheduler.ExponentialLR(vis_reweight_optimizer, gamma)




            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            # 四、train + logging + save

            val_min_acc = 0
            val_min_loss = float('inf')
            csv_overwrite_flag_this_train=csv_overwrite_flag
            # epoch自适应
            acc_not_increase_flag_list = {'train':0,'val':0} 
            max_acc_list = {'train':0,'val':0} 

            # epoch自适应
            for epoch in range(start_epoch, end_epoch*100):
                start_time = time.time()
                list_acc_loss = []
                list_acc_loss.append(epoch)
                print("epoch:{}".format(epoch)) 
                for train_or_val_phase in ['val', 'train']:
                    dataloader = dataloaders[train_or_val_phase]
                    (dict_loss_acc_epoch, confusion_mat_epoch)  = train(
                            train_or_val_phase, 
                            model_list,  optimizer_list,  opt_scheduler_list, 
                            vis_reweight_model, vis_reweight_optimizer, vis_reweight_scheduler,

                            criterion_ce,  
                            criterion_spc, loss_spc_weight,  
                            criterion_triplet, loss_triplet_weight,  

                            criterion_huber,loss_vis_huber_weight,
                            criterion_infonce,loss_vis_infonce_weight,
                            criterion_kl,loss_vis_kl_weight,

                            dataloader, 
                            mask_num, mask_len, 

                            scaler, 
                            rank, world_size, 

                            epoch, 
                            save_path_csv_batch=save_dir_csv_batch_this_sub,  
                            csv_fname=csv_fname, 
                            overwrite_flag=csv_overwrite_flag, 
                            parameter_info_thorough=parameter_str_short, 
                            
                            pic_emb=pic_emb,
                            
                                )
                    # 参数 mask_num, mask_len, device, accelerator, save_path_csv_batch,  
                    # csv_fname, overwrite_flag, parameter_info_thorough,  并未使用
                            
                    list_loss_acc_epoch=dict_tolist_loss_acc(dict_loss_acc_epoch)
                    list_acc_loss+=list_loss_acc_epoch
                    
                    # # 用于save best模型 ---- 只保存SCT模型即可，其余6个均不保存
                    # if train_or_val_phase=='val':
                    # # if epoch%1 ==0:#由于我还没处理测试集，所以每10个epoch存一下，方便日后寻找最优模型（轮流load，看测试集finetune的准确率）
                    #     val_loss=dict_loss_acc_epoch['loss_total']
                    #     val_acc=dict_loss_acc_epoch['accuracy']
                    #     # 保存当前的 及 最好的测试性能的 模型和优化器
                    #     # val_loss = list_acc_loss[pos_val_loss]#!!!这里一定要注意val_loss在list_acc_loss中的位置!!!

                        
                    #     # acc和loss各存一个
                    #     best_acc_or_loss = 'loss'
                    #     val_min_loss = save_checkpoint_best_and_current(
                    #                         save_dir_ckpt_this_sub, checkpoint_fname_rear,
                    #                         epoch, model, optimizer,
                    #                         val_loss, val_min_loss, 
                    #                         best_acc_or_loss,
                    #                         accelerator,
                    #                         whether_remove_old_model=True)

                    #     # acc和loss各存一个
                    #     best_acc_or_loss = 'acc'
                    #     val_min_acc = save_checkpoint_best_and_current(
                    #                         save_dir_ckpt_this_sub, checkpoint_fname_rear,
                    #                         epoch, model, optimizer,
                    #                         val_acc, val_min_acc, 
                    #                         best_acc_or_loss,
                    #                         accelerator,
                    #                         whether_remove_old_model=True)
                    

                    # print
                    step=None
                    if epoch == start_epoch:#打印1个epoch的用时
                        time_duration=time.time()-start_time
                    else:
                        time_duration
                    # 新加入了batch_num，是epoch内查看进度用的
                    print_loss_acc_dict(dict_loss_acc_epoch, train_or_val_phase, epoch, step, 
                                        batch_num=len(dataloader), time_duration=time_duration)
                    # # 保存当前train_or_val_phase的confusion矩阵
                    # save_confusion_matrix(save_path_confusion_mat_this_para, epoch, train_or_val_phase, confusion_mat_epoch_nomask,  'nomask')
                    # save_confusion_matrix(save_path_confusion_mat_this_para, epoch, train_or_val_phase, confusion_mat_epoch_mask,  'mask')

                    # epoch自适应
                    acc_this_epoch  = dict_loss_acc_epoch['accuracy']
                    if acc_this_epoch > max_acc_list[train_or_val_phase]:
                        max_acc_list[train_or_val_phase] = acc_this_epoch
                        acc_not_increase_flag_list[train_or_val_phase]  = 0
                    else:
                        acc_not_increase_flag_list[train_or_val_phase]  += 1            

                # 保存loss 总的准确率 每一类准确率（train和test）
                save_acc_loss(
                    save_dir_csv_this_sub, csv_fname,
                    list_acc_loss,
                    csv_overwrite_flag_this_train,
                    parameter_info_thorough=parameter_str_short)
                #*-重要-*  csv_overwrite_flag_this_train
                csv_overwrite_flag_this_train=0#保存过1次后，就置零

                # epoch自适应
                # 假如训练集验证集准确率都不再增长，则终止训练
                if epoch>end_epoch:
                    if acc_not_increase_flag_list['train']>20 or acc_not_increase_flag_list['val']>30:
                        print(Fore.RED+'Training ended at epoch',epoch)
                        break

            print('foldNO {} done!'.format(foldNO))
        print('{} {} done!'.format(sub_list, model_str))
    print('已完成：{} done!'.format(parameter_str_short))












