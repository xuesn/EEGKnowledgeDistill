import torch
import os
import pandas as pd
import copy
import numpy as np
import imageio 

# 98上sklearn报错
# 未使用sklearn的，用的自己写的confusion_matrix
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
#例： OUT：
# [[66 55]
#  [35 94]]
#               precision    recall  f1-score   support

#          0.0       0.65      0.55      0.59       121
#          1.0       0.63      0.73      0.68       129

#     accuracy                           0.64       250
#    macro avg       0.64      0.64      0.64       250
# weighted avg       0.64      0.64      0.64       250


# 模型 类别预测矩阵 文件文件夹名~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_csv_fname(parameter_info_short):
    csv_prefix='loss'
    csv_fname = csv_prefix +\
        '_'+parameter_info_short +  \
        '.csv'
    return csv_fname

def get_checkpoint_fname_rear(parameter_info_short):
    checkpoint_fname_rear = '_'+parameter_info_short+\
        '_checkpoint.pt'
    return checkpoint_fname_rear

#保存~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#准确率和损失值--------------------------------------------------
def save_acc_loss(
        save_path_csv, csv_fname,
        save_list_row,
        overwrite_flag,
        parameter_info_thorough):
    
    # print(save_path_csv)
    # print(csv_fname)
    # assert False

    #新建文件夹
    if not os.path.exists(save_path_csv):
        os.makedirs(save_path_csv)
        print('已新建文件夹：',save_path_csv, ' created!')
    full_path = os.path.join(save_path_csv, csv_fname)
    # 如果文件尚未被创建 或者 overwrite_flag==1，则先新建再写入
    if (not os.path.exists(full_path)) or (overwrite_flag == 1):
        columns = ['epoch',


                   'val loss_total', 'val loss_ce',  
                   'val loss_vis_huber',  'val loss_vis_infonce',
                   'val loss_vis_kl',  'val accuracy', 
                   'val loss_spc', 'val loss_triplet',  

                   'train loss_total', 'train loss_ce',  
                   'train loss_vis_huber',  'train loss_vis_infonce',
                   'train loss_vis_kl',  'train accuracy', 
                   'train loss_spc', 'train loss_triplet',  
                   ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(full_path, index=False)
        file_info = [full_path,parameter_info_thorough]  # 存一下文件名，包含文件信息
        df = pd.DataFrame(data=file_info)
        df.to_csv(full_path,  mode='a',
                  header=False, index=False)
    df = pd.DataFrame(data=[save_list_row])  # 注意list要多加一层'[·]'
    df.to_csv(full_path, mode='a',
              header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
    return

def save_acc_loss_batch(
        save_path_csv_batch, csv_fname,
        save_list_row,
        overwrite_flag,
        parameter_info_thorough_finetune):
    #新建文件夹
    if not os.path.exists(save_path_csv_batch):
        os.makedirs(save_path_csv_batch)
        print('已新建文件夹：',save_path_csv_batch, ' created!')
    full_path = os.path.join(save_path_csv_batch, csv_fname)
    # 如果文件尚未被创建 或者 overwrite_flag==1，则先新建再写入
    if (not os.path.exists(full_path)) or (overwrite_flag == 1):
        columns = ['epoch','batch','train_or_val_phase',
                  
                   'loss_total', 'loss_ce',  
                   'loss_vis_huber',  'loss_vis_infonce',
                   'loss_vis_kl',  'accuracy', 
                   'loss_spc', 'loss_triplet',  
                ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(full_path, index=False)
        file_info = [full_path,parameter_info_thorough_finetune]  # 存一下文件名，包含文件信息
        df = pd.DataFrame(data=file_info)
        df.to_csv(full_path,  mode='a',
                  header=False, index=False)
    df = pd.DataFrame(data=[save_list_row])  # 注意list要多加一层'[·]'
    df.to_csv(full_path, mode='a',
              header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
    return

#checkpoint--------------------------------------------------
def save_checkpoint(
        save_path_checkpoint, checkpoint_fname_rear,
        epoch, model, optimizer,
        val_acc_or_loss,
        checkpoint_fname_prefix,
        accelerator,
        whether_remove_old_model
        ):
    #新建文件夹
    if not os.path.exists(save_path_checkpoint):
        os.makedirs(save_path_checkpoint)
        print('已新建文件夹：',save_path_checkpoint, ' created!')

    sameparameter_list = os.listdir(save_path_checkpoint)
    sameparameter_list = [
        fn for fn in sameparameter_list if fn.endswith(checkpoint_fname_rear)]
    # 注意，保存模型之后再删除原来的
    target_list = [
        fn for fn in sameparameter_list if fn.startswith(checkpoint_fname_prefix)]
    # 保存模型和优化器
    ckpt_fullpath = os.path.join(save_path_checkpoint,
                        checkpoint_fname_prefix+('%.4f' % val_acc_or_loss) +
                        '_'+'epoch-' + str(epoch) +
                        '_'+checkpoint_fname_rear)
    unwrapped_model = accelerator.unwrap_model(model)
    # 以checkpoints形式保存模型的相关数据
    torch.save({
        'epoch': epoch,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc_or_loss': val_acc_or_loss,
    }, ckpt_fullpath)
    # 预训练时不能移除旧模型
    if whether_remove_old_model:
        # 注意，保存模型之后再删除原来的
        # 一般只保存了1个，故只删1个，也可以避免万一的情况删掉了所有的
        assert (len(target_list) == 0) | (
            len(target_list) == 1), '！！！之前保存了不止一个模型！！！'
        if len(target_list) == 1:
            os.remove(os.path.join(save_path_checkpoint, target_list[0]))
    return

# 保存当前的 及 最好的测试性能的 模型和优化器
def save_checkpoint_best_and_current(
        save_path_checkpoint, checkpoint_fname_rear,
        epoch, model, optimizer,
        val_acc_or_loss, val_best_acc_or_loss, 
        best_acc_or_loss,
        accelerator,
        whether_remove_old_model
    ):
    #更新最优val准确率或损失值
    flag_update_best_ckpt=False
    if best_acc_or_loss=='acc':
        if val_acc_or_loss>val_best_acc_or_loss:
            val_best_acc_or_loss=val_acc_or_loss
            flag_update_best_ckpt=True
    elif best_acc_or_loss=='loss':
        if val_acc_or_loss<val_best_acc_or_loss:
            val_best_acc_or_loss=val_acc_or_loss
            flag_update_best_ckpt=True
    else:
        assert False," best_acc_or_loss can only be 'acc' or 'loss' "

    # 在外部控制为好，这里先注释看看会不会出问题
    # #分布式时，只保存一个ckpt
    # if not accelerator.is_main_process:
    #     return val_best_acc_or_loss

    #新建文件夹
    if not os.path.exists(save_path_checkpoint):
        os.makedirs(save_path_checkpoint)
        print('已新建文件夹：',save_path_checkpoint, ' created!')
    # 1、保存 最好的测试性能的 模型和优化器
    if flag_update_best_ckpt:
        best_checkpoint_fname_prefix='BEST-val_'+best_acc_or_loss+'-'
        save_checkpoint(
                save_path_checkpoint, checkpoint_fname_rear,
                epoch, model, optimizer,
                val_best_acc_or_loss,
                best_checkpoint_fname_prefix,
                accelerator,
                whether_remove_old_model
                )
    #############################################
    # 2、保存 当前epoch的 模型和优化器
    current_checkpoint_fname_prefix='Current-val_'+best_acc_or_loss+'-'
    save_checkpoint(
        save_path_checkpoint, checkpoint_fname_rear,
        epoch, model, optimizer,
        val_acc_or_loss,
        current_checkpoint_fname_prefix,
        accelerator,
        whether_remove_old_model
        )
    return val_best_acc_or_loss  #24-05-10 发现这里返回的是val_acc_or_loss  修改为val_best_acc_or_loss

#类别预测矩阵--------------------------------------------------
# 行：模型预测的类别  列：真实类别
def save_confusion_matrix(save_path_confusion_mat,epoch,train_or_val_phase,confusion_mat, whether_data_masked_str):
    #1、保存confusion-mat为csv表格
    fname_csv='epoch'+str(epoch).zfill(3)+'_'+train_or_val_phase+'.csv'
    save_dir_csv=os.path.join(save_path_confusion_mat, whether_data_masked_str+'_'+'csv')
    # 创建文件夹
    if not os.path.exists(save_dir_csv):
        os.makedirs(save_dir_csv)
        print('已新建文件夹：',save_dir_csv, ' created!')
    save_path_csv=os.path.join(save_dir_csv, fname_csv)
    np.savetxt(save_path_csv,
                confusion_mat.astype('int'), delimiter=',')
    # print('已保存：',save_path_csv,' saved!')
    #2、保存confusion-mat为jpg 绿色渐变图像
    fname_jpg='epoch'+str(epoch).zfill(3)+'_'+train_or_val_phase+'.jpg'
    save_dir_jpg=os.path.join(save_path_confusion_mat, whether_data_masked_str+'_'+'jpg')
    # 创建文件夹
    if not os.path.exists(save_dir_jpg):
        os.makedirs(save_dir_jpg)
        print('已新建文件夹：',save_dir_jpg, ' created!')
    save_path_jpg=os.path.join(save_dir_jpg,fname_jpg)
    #归一化到0~255  采用白色到绿色线性渐变
    normed_confusion_mat=255-np.floor((confusion_mat-np.min(confusion_mat))/(np.max(confusion_mat)-np.min(confusion_mat))*255)
    normed_confusion_mat=np.expand_dims(normed_confusion_mat, axis=2)#reshape出第3维
    green_img=np.concatenate((normed_confusion_mat,np.ones(normed_confusion_mat.shape)*255,normed_confusion_mat),axis=2).astype('uint8')
    more_pixel_green_img=np.repeat(green_img, 20, axis=0)
    more_pixel_green_img=np.repeat(more_pixel_green_img, 20, axis=1)
    imageio.imwrite(save_path_jpg, more_pixel_green_img)
    # print('已保存：',save_path_jpg,' saved!')
    return


