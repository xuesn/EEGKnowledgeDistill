import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'DETAIL'

import time
import numpy as np
import torch


# loss计算 
def get_acc_confusion_mat(prob, label_onehot):
    #分类正确数计算
    # 返回最大值的位置，注意是从0开始的
    pred = np.argmax(prob.cpu().detach().numpy(),  axis=1)
    label = np.argmax(label_onehot.cpu().detach().numpy(),  axis=1)
    # 由于是多标签，这里要改了
    label_onehot = label_onehot.cpu().detach().numpy()
    label_onehot_row = label_onehot.reshape([1,  -1])
    # 等差数列
    batch_sample_num,  class_num = prob.shape
    arithmetic_sequence = np.arange(batch_sample_num)*class_num
    row_position_pred = arithmetic_sequence+pred
    # 只要是多标签中的1个，就视为预测正确
    right_num_batch = float(sum(label_onehot_row[0, row_position_pred] == 1))
    accuracy_batch = right_num_batch / batch_sample_num
    #记录每一类的预测confusion矩阵 （注意不适合多类别的情况）
    class_num=label_onehot.shape[-1]
    confusion_mat_batch=np.zeros([class_num, class_num],  dtype=np.int32)
    for i,  pred_i in enumerate(pred):
        confusion_mat_batch[pred_i,  label[i]] += 1  # 行：模型预测的类别  列：真实类别
    return (right_num_batch, accuracy_batch, confusion_mat_batch)


def get_loss_ce(prob, label_onehot,criterion_ce):
    # CE ----
    loss_ce = criterion_ce(prob,  label_onehot) 
    return loss_ce    

  
def gather_features_supcon(
        eeg_features, 
        local_loss=False, 
        rank=0, 
        world_size=1, 
        ):
    gathered_eeg_features = [torch.zeros_like(eeg_features) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_eeg_features,  eeg_features)
    if not local_loss:
        # ensure grads for local rank when all_* features don't have a gradient
        gathered_eeg_features[rank] = eeg_features
    all_eeg_features = torch.cat(gathered_eeg_features,  dim=0)
    #拼成batch-size*world-size行 的特征
    return all_eeg_features
def get_loss_contrast(rank, world_size, 
              emb_norm, label_classNO,
              criterion_spc, criterion_triplet,):
    '''
    # 多卡对比学习 ----
    all_eeg_features=gather_features_supcon(
            emb_norm,
            local_loss=False,
            rank=rank,
            world_size=world_size,)
    all_batch_y_classNO=gather_features_supcon(
            label_classNO,
            local_loss=False,
            rank=rank,
            world_size=world_size,)            
    # SPC ----
    loss_spc = criterion_spc(all_eeg_features, all_batch_y_classNO, rank) 
    # Triplet ----
    loss_triplet ,_ = criterion_triplet(all_eeg_features, all_batch_y_classNO, rank, ) # margin, squared=False可以修改 
    '''
    # 单卡对比学习 ----
    # SPC ----
    loss_spc = criterion_spc(emb_norm, label_classNO, rank)  
    # Triplet ----
    loss_triplet ,_ = criterion_triplet(emb_norm, label_classNO, rank, ) # margin, squared=False可以修改
    # loss_triplet, fraction_postive_triplets
    return loss_spc, loss_triplet 
    


def get_loss_acc_two_visual_linear(rank, world_size, 
        emb_norm, prob, 
        concatenated_eeg, concatenated_label_onehot, label_classNO, 
        emb_vis,linear_out,
        criterion_ce,  
        criterion_spc, loss_spc_weight,   
        criterion_triplet, loss_triplet_weight,  
        criterion_huber,loss_vis_huber_weight,
        criterion_infonce,loss_vis_infonce_weight,
        criterion_kl,loss_vis_kl_weight, 
        train_or_val_phase):
    # 先初始化为0，避免报错
    (loss_spc, loss_triplet, )=(
        torch.tensor(0),torch.tensor(0),)
    (loss_vis_huber, loss_vis_infonce, loss_vis_kl)=(
        torch.tensor(0),torch.tensor(0),torch.tensor(0))

    # # sub08-fold0-不用if layNO == 0:的话这俩好像都变差，但估计是随机性的，不会一直差
    # (loss_vis_huber, loss_vis_infonce, loss_vis_kl)=(
    #     torch.tensor(0).half().to(rank),torch.tensor(0).half().to(rank),torch.tensor(0).half().to(rank))
    # (loss_vis_huber, loss_vis_infonce, loss_vis_kl)=(
    #     emb_vis[0,0]*0,emb_vis[0,0]*0,emb_vis[0,0]*0)
    # 因为emb_vis = emb_vis.squeeze().half().to(rank)

    # 要约定好mask在前：concatenated_eeg = torch.cat([masked_eeg,eeg], dim = 0, )
    whole_sample_num = concatenated_eeg.shape[0]
    half_sample_num = int(whole_sample_num/2)

    ##### 一、
    # ce-loss 准确率 confusion-mat
    loss_ce = get_loss_ce(prob, concatenated_label_onehot,criterion_ce)
    (right_num, accuracy_batch, 
        confusion_mat_batch) = get_acc_confusion_mat(prob, concatenated_label_onehot)

    ##### 二、
    if train_or_val_phase == 'train':
        # 对比的loss就不区分是否mask了，因为mask后的也加入特征对比了
        if loss_spc_weight > 0 or loss_triplet_weight > 0:
            loss_spc,loss_triplet = get_loss_contrast(rank, world_size, 
                emb_norm, label_classNO,
                criterion_spc, criterion_triplet)
            if loss_spc>999 and torch.isnan(loss_spc):
                loss_spc = 0*emb_norm
            if loss_triplet>999 and torch.isnan(loss_triplet):
                loss_triplet = 0*emb_norm

        ##### 三、
        # AE的loss可以区分


        ##### 四、
        # visual的loss可以区分
        # 1、EMBEDDING
        # list以后再用，先看看有无提升
        loss_vis_huber_list = []
        loss_vis_infonce_list = []
        loss_vis_kl_list = []
        
        # for layNO, linear_out in enumerate(proj_emb_eeg_list):
            # emb_vis = emb_vis_multi_layer[:,layNO,:]
        layNO = 0
        if loss_vis_huber_weight > 0 :
            loss_vis_huber_list.append(criterion_huber(linear_out, emb_vis))
            if loss_vis_huber_list[-1]>9999 and torch.isnan(loss_vis_huber_list[-1]>9999):
                loss_vis_huber_list[-1]= 0*linear_out 
            if layNO == 0:
                loss_vis_huber = criterion_huber(linear_out, emb_vis)
            else:
                loss_vis_huber+=loss_vis_huber_list[-1]
        #
        if loss_vis_infonce_weight > 0 :
            loss_vis_infonce_list.append(criterion_infonce(linear_out, emb_vis))
            if loss_vis_infonce_list[-1]>9999 and torch.isnan(loss_vis_infonce_list[-1]):
                loss_vis_infonce_list[-1]= 0*linear_out 
            if layNO == 0:
                loss_vis_infonce = criterion_infonce(linear_out, emb_vis)
            else:
                loss_vis_infonce+=loss_vis_infonce_list[-1]
        #
        if loss_vis_kl_weight > 0 :
            loss_vis_kl_list.append(criterion_kl(linear_out, emb_vis))
            if loss_vis_kl_list[-1]>9999 and torch.isnan(loss_vis_kl_list[-1]):
                loss_vis_kl_list[-1]= 0*linear_out 
            if layNO == 0:
                loss_vis_kl = criterion_kl(linear_out, emb_vis)
            else:
                loss_vis_kl+=loss_vis_kl_list[-1]

        
    elif train_or_val_phase == 'val':
        loss_spc = 0*emb_norm.sum()
        loss_triplet = 0*emb_norm.sum()
        loss_vis_huber = 0*emb_norm.sum()
        loss_vis_infonce = 0*emb_norm.sum()
        loss_vis_kl = 0*emb_norm.sum()
    # loss加和
    # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss. 
    unused_parameter_tensor= 0* prob.sum() + 0* emb_norm.sum()  #230915新加
    # !!!

    loss_total = (loss_ce 
        + loss_spc_weight*loss_spc 
        + loss_triplet_weight*loss_triplet

        + loss_vis_huber_weight*loss_vis_huber
        + loss_vis_infonce_weight*loss_vis_infonce
        + loss_vis_kl_weight*loss_vis_kl

        + unused_parameter_tensor)
    # unused_parameter_tensor
    


    return (loss_total, 
            loss_ce, 
            loss_spc, loss_triplet, 
            loss_vis_huber, loss_vis_infonce, loss_vis_kl, 
            right_num, accuracy_batch, 
            confusion_mat_batch,)

