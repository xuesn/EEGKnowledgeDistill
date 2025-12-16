

import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'DETAIL'

import time
import numpy as np
import torch

from save.utils_save import save_acc_loss_batch
from save.utils_loss_acc_dict import *   # 注意未放在train路径下，因为main中也用得到

from .utils_loss import *  


# train / val
def train(
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
    save_path_csv_batch,  csv_fname, 
    overwrite_flag, 
    parameter_info_thorough,  

    pic_emb
        ): 
    #获取类别数，以初始化confusion_mat
    eeg, label_onehot, imgNO= dataloader.dataset.__getitem__(1)
    timepoint_num,  electrode_num = eeg.shape
    class_num = label_onehot.shape[0]
    confusion_mat_epoch = np.zeros(
        [class_num,  class_num],  dtype=np.int32)  

    #训练 or 测试
    if train_or_val_phase == 'train':
        for model in model_list:
            model.train()
    elif (train_or_val_phase == 'val') or (train_or_val_phase == 'test'):
        for model in model_list:
            model.eval()
    else:
        assert 'train_or_val_phase can only be '+'train'+' or ' + 'val'+' or ' + 'test'
    dict_loss_acc_epoch=initial_loss_acc_dict()
    sample_num = len(dataloader.dataset)
    # print('sample_num:{}'.format(sample_num)) 
    batch_num = len(dataloader)
    # print('batch_num:{}'.format(batch_num)) 
    for step,  ( eeg, label_onehot,  imgNO ) in enumerate(dataloader):
        # break
        #保存每个batch的loss acc
        dict_loss_acc_batch=initial_loss_acc_dict()
        start_time=time.time()
        with torch.set_grad_enabled(train_or_val_phase == 'train'):
            eeg = eeg.to(rank)
            sample_num_this_batch = eeg.shape[0]
            concatenated_eeg = torch.cat([eeg,eeg], dim = 0, )

            label_onehot = label_onehot.to(rank)
            concatenated_label_onehot = torch.cat([label_onehot,label_onehot], dim = 0,)
            # _,  label_classNO = torch.max(
            #     concatenated_label_onehot,  dim=1)  # 有监督对比不用one-hot     
            # 构造无监督对比的label_classNO
            label_classNO = torch.arange(sample_num_this_batch)
            for i in range(1):
                label_classNO = torch.cat([label_classNO,torch.arange(sample_num_this_batch)] )      
            label_classNO = label_classNO.to(rank)
            # eeg和mask_eeg一起过模型 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
            # 前向过程(model + loss)开启 autocast
            with torch.cuda.amp.autocast():
                dec_out_normed, emb_norm = model_list[0](concatenated_eeg)
                # 开始多层映射
                proj_emb_eeg_list = []
                last_emb_eeg = dec_out_normed
                for model_proj in model_list[1:-1]:
                    if len(proj_emb_eeg_list)<2:
                        proj_emb_eeg_list.append( model_proj(last_emb_eeg) )
                    else:  # 加残差
                        proj_emb_eeg_list.append( model_proj(last_emb_eeg+proj_emb_eeg_list[-2]) )
                    last_emb_eeg = proj_emb_eeg_list[-1]
                prob = model_list[-1](last_emb_eeg)


                concatenated_imgNO = np.concatenate([imgNO, imgNO],axis=0) 
                emb_vis = torch.tensor(pic_emb[concatenated_imgNO])
                emb_vis = emb_vis.squeeze().half().to(rank)

                '''
                以后逐层学习时很可能要用这个
                dim = 1 可能是单层学习或逐层学习的兼容写法
                # print(emb_vis_multi_layer.shape) # (bs, 1, 10, 768)
                emb_vis_multi_layer = emb_vis_multi_layer.squeeze(dim=1).half().to(rank)
                # print(emb_vis_multi_layer.shape) # (bs, 10, 768)
                '''

                if vis_reweight_model is not None:
                    reweight_emb_vis = vis_reweight_model(emb_vis)
                    emb_vis = reweight_emb_vis.squeeze().half()


                # 计算loss和acc
                (loss_total, 
                    loss_ce, 
                    loss_spc, loss_triplet, 
                    loss_vis_huber, loss_vis_infonce, loss_vis_kl, 
                    right_num, accuracy_batch, 
                    confusion_mat_batch,)=get_loss_acc_two_visual_linear(rank, world_size, 
                        emb_norm, prob, 
                        concatenated_eeg, concatenated_label_onehot, label_classNO, 
                        emb_vis,last_emb_eeg,
                        criterion_ce,  
                        criterion_spc, loss_spc_weight,   
                        criterion_triplet, loss_triplet_weight, 
                        criterion_huber,loss_vis_huber_weight,
                        criterion_infonce,loss_vis_infonce_weight,
                        criterion_kl,loss_vis_kl_weight, 
                        train_or_val_phase)

            #模型参数更新    # 每个batch更新一次参数
            if train_or_val_phase == 'train':
                for optimizer in optimizer_list:
                    optimizer.zero_grad()
                if vis_reweight_model is not None:
                    vis_reweight_optimizer.zero_grad()

                scaler.scale(loss_total).backward()
                for optimizer in optimizer_list:
                    scaler.step(optimizer)
                if vis_reweight_model is not None:
                    scaler.step(vis_reweight_optimizer)
                        
                # 看是否要增大scaler
                scaler.update()
            

        #confusion_mat 不每个batch保存了，内存也占的大
        # 行：模型预测的类别  列：真实类别
        confusion_mat_epoch+=confusion_mat_batch
        #保存每个batch的loss acc
        # SPECIAL！！concat后要除2
        right_num = int(right_num/2)
        dict_loss_acc_batch=tensor_loss_acc_dict(dict_loss_acc_batch,    
                        loss_total, loss_ce, 
                        loss_vis_huber, loss_vis_infonce, loss_vis_kl, 
                        right_num,
                        loss_spc,  
                        loss_triplet)                    
        list_loss_acc_batch=[epoch, step, train_or_val_phase]
        list_loss_acc_batch += dict_tolist_loss_acc(dict_loss_acc_batch)
        #
        # save_acc_loss_batch(save_path_csv_batch,  
        #     csv_fname, 
        #     list_loss_acc_batch, 
        #     overwrite_flag, 
        #     parameter_info_thorough)
        overwrite_flag=0#保存过1次后，就置零

        #打印信息
        time_duration = time.time() - start_time
        #
        # if step%2 ==0:
        if step%10_000 ==9_999:
            print_loss_acc_dict(dict_loss_acc_batch,  train_or_val_phase,  epoch,  step, batch_num, time_duration)

        #加入epoch的loss里
        dict_loss_acc_epoch=add_loss_acc_dict(dict_loss_acc_epoch, dict_loss_acc_batch)

    # 每个epoch的学习率调节
    if train_or_val_phase == 'train':
        # gamma=1，其实暂未做学习率控制
        for opt_scheduler in opt_scheduler_list:
            opt_scheduler.step()
        if vis_reweight_scheduler is not None:
            vis_reweight_scheduler.step()

    # loss除以batch数，因为loss是对batch内的样本数取过平均的，而不是每个样本的loss相加
    dict_loss_acc_epoch=divide_loss_acc_dict(dict_loss_acc_epoch, batch_num, sample_num)

    return dict_loss_acc_epoch,  confusion_mat_epoch
    



