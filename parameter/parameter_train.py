import logging
from torch import nn
import numpy as np

from .loss import SupervisedContrastiveLoss_myrevised, BatchAllTripletLoss,InfoNCE,KLDivLoss
from utils.utils import main_dir


def parameter_train():
    # 随机种子
    seed = 50
    # 迭代
    start_epoch = 0
    end_epoch = 100
    # 续跑参数 
    load_latest_model = 1
    csv_overwrite_flag = 0
    # batch_size
    train_batch_size = 32  # 15S 21P
    train_batch_size = 125 # 22G
    
    
    # # ######测试用
    # train_batch_size=2        
    val_batch_size=int(train_batch_size*2)#测试时无梯度，可以大一点    

    return seed, start_epoch, end_epoch, load_latest_model, csv_overwrite_flag, train_batch_size, val_batch_size


def parameter_regularize():
    #dropout
    # dropout_trans_list=[0,0.25,0.5,0.75]
    # dropout_conv_list=[0,0.25,0.5,0.75]
    # dropout_fc_list=[0,0.25,0.5,0.75]
    # weight_decay_list = [0.005,0.001,0.0005,0.0001,0.00005]
    dropout_trans=0.1
    dropout_conv=0.25
    dropout_fc=0
    weight_decay = 0
    logging.info('weight_decay:{},dropout_trans:{},dropout_conv:{},dropout_fc:{},'.format(weight_decay,dropout_trans,dropout_conv,dropout_fc))
    return dropout_trans, dropout_conv, dropout_fc, weight_decay   
# dropout_trans, dropout_conv, dropout_fc, weight_decay = parameter_regularize()


def parameter_optimizer():
    # 优化器参数
    # 平滑常数\beta_1和\beta_2
    beta_1 = 0.9
    beta_2 = 0.999
    # eps加在分母上防止除0
    eps=1e-08
    # 学习率
    # learn_rate_list=[0.00001,0.0001,0.001]
    learn_rate = 1e-5*100 
    logging.info('learn_rate:{}'.format(learn_rate))
    gamma=1  # opt_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    return beta_1, beta_2, eps, learn_rate, gamma 




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
                  'resnet50', 'resnet101', 'vit_b_16']+\
                  ['clip-vit-base-patch32','clip-vit-large-patch14']
    feat_dim_dict={}
    feat_dim_dict['alex']=9216
    feat_dim_dict['resnet18']=512
    feat_dim_dict['resnet34']=512
    feat_dim_dict['resnet50']=2048
    feat_dim_dict['resnet101']=2048
    feat_dim_dict['vit_b_16']=768    
    #
    # CLIP-ViT proj后的  效果很差
    # feat_dim_dict['clip-vit-base-patch32']=512
    # feat_dim_dict['clip-vit-large-patch14']=768
    # CLIP-ViT proj前的
    feat_dim_dict['clip-vit-base-patch32']=768
    feat_dim_dict['clip-vit-large-patch14']=1024

    #
    #   ['clip-vit-base-patch32_dim512.npy','clip-vit-large-patch14_dim768.npy']
    if vis_emb_type == 'image':
        pic_emb_fname = vis_emb_str+'_dim'+str(feat_dim_dict[vis_emb_str])+'.npy'
    elif vis_emb_type == 'label':
        pic_emb_fname = vis_emb_str+'_dim'+str(feat_dim_dict[vis_emb_str])+'_label.npy'
    import os
    pic_emb_path = os.path.join(pic_emb_dir,dataset_dir,pic_emb_fname)
    pic_emb = np.load(pic_emb_path)
    
    return pic_emb,  feat_dim_dict[vis_emb_str]


def parameter_loss(dataset_str):
    # mask-AE重建
    # loss_mse_mask_weight_list= [1000, 100,10, 1,1e-1,1e-2,1e-3,]
    # delta_list = [0.25,0.5,0.75,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]   
    dlt_ae = 0.4
    loss_ae_weight = 0
    
    # 对比
    # InfoNCE
    # temperature_list=[0.1, 0.25, 0.5, 0.75]
    temp_eeg = 0.1
    # loss_spc_weight_list= [100, 10, 1, 0.1, 0.01, 0.001,]
    loss_spc_weight = 0  
    loss_spc_weight = 1
    
    # triplet
    # margin_list=[0, 0.5,  1,  2,  5]
    margin_eeg = 1  # 15S  21P
    margin_eeg = 0  # 22G 

    # loss_triplet_weight_list=  [100, 10, 1, 0.1, 0.01, 0.001,]
    loss_triplet_weight = 0
    loss_triplet_weight = 1
    
    
    # 视觉监督
    delta_vis = 0.5   # 15S 21P
    delta_vis = 0.4   # 22G

    temperature_vis = 0.1
    loss_vis_huber_weight = 50   # 15S 21P
    loss_vis_huber_weight = 100  # 22G
    # loss_vis_huber_weight = 0 
    loss_vis_infonce_weight = 0
    loss_vis_kl_weight = 0
    # vis_emb_type_list=['image','label']  # image：每个图片的特征  label：所有同类图片的特征均值
    vis_emb_type = 'image'
    
    # 选择要使用的视觉特征
    vis_emb_str = 'alex'
    vis_emb_str = 'resnet18'
    vis_emb_str = 'resnet34'
    vis_emb_str = 'resnet50'
    vis_emb_str = 'resnet101'
    vis_emb_str = 'vit_b_16'
    vis_emb_str = 'clip-vit-base-patch32'
    vis_emb_str = 'clip-vit-large-patch14'

    # vis_reweight_type_list=['token','crop75','crop50',]
    # vis_reweight_type_list=['layer3','layer10','layer5',]
    # vis_layer_list_list =  [
    #     [9],
    #     [8,9],
    #     [7,8,9],
    #     [7,9],
    #     [6,7,8,9],
    #     [5,7,9],
    #     [5,6,7,8,9],
    #     ]

    vis_reweight_type = ""
    vis_layer_list = [0]
    # 选择vis_emb
    if vis_reweight_type =="token":
        vis_emb_num = 197 # vit-b-16
        pic_emb_dir = main_dir + '/visual_embedding_new0713_token/'
    elif vis_reweight_type =="layer10":
        vis_emb_num = 10 # vit-b-16
        pic_emb_dir = main_dir + '/visual_embedding_new0716_layer10/'
    elif vis_reweight_type =="layer5":
        vis_emb_num = 5 # vit-b-16
        pic_emb_dir = main_dir + '/visual_embedding_new0716_layer5/'
    elif vis_reweight_type =="layer3":
        vis_emb_num = 3 # vit-b-16
        pic_emb_dir = main_dir + '/visual_embedding_new0716_layer3/'
    elif vis_reweight_type =="crop75":
        vis_emb_num = 5 # vit-b-16
        pic_emb_dir = main_dir + '/visual_embedding_new0715_crop0p75/'
    elif vis_reweight_type =="crop50":
        vis_emb_num = 5 # vit-b-16
        pic_emb_dir = main_dir + '/visual_embedding_new0715_crop0p5/'
    elif vis_reweight_type =="":
        vis_emb_num = 1
        pic_emb_dir = main_dir + '/data/visual_embedding/'
    
    pic_emb, vis_emb_dim = get_pic_emb(pic_emb_dir, dataset_str, vis_emb_str, vis_emb_type)
    pic_emb = pic_emb[:,np.newaxis,:] 


    # flag_layerwise_or_reweight : 'layerwise' or 'reweight'
    flag_layerwise_or_reweight = 'layerwise'  # 逐层学习（包括只有普通的最后一层学习）
    # flag_layerwise_or_reweight = 'reweight'  # 尝试 token/layer加权  crop估计没有用不试了

    vis_layer_str = ''
    if flag_layerwise_or_reweight == 'layerwise':
        vis_emb_num = len(vis_layer_list)
        pic_emb = pic_emb[:,vis_layer_list,:]
        for vis_layerNO in vis_layer_list:
            vis_layer_str+=str(vis_layerNO)


    # loss
    criterion_ce = nn.CrossEntropyLoss()  
    #
    # criterion_mask =  nn.MSELoss(reduction='mean')  
    criterion_ae =  nn.HuberLoss(reduction='mean', delta=dlt_ae)
    #
    criterion_spc = SupervisedContrastiveLoss_myrevised(temp_eeg)  # 该loss输入标签不能为one-hot  
    criterion_triplet = BatchAllTripletLoss(margin_eeg)
    #    
    criterion_huber = nn.HuberLoss(reduction='mean', delta=delta_vis)
    criterion_infonce = InfoNCE(temperature=temperature_vis, reduction='mean', negative_mode='unpaired')
    criterion_kl = KLDivLoss()

    return (dlt_ae,loss_ae_weight,
    temp_eeg,loss_spc_weight,margin_eeg,loss_triplet_weight,
    delta_vis, temperature_vis, loss_vis_huber_weight, loss_vis_infonce_weight, loss_vis_kl_weight, vis_emb_str, vis_emb_type, vis_reweight_type, vis_layer_str,   
    pic_emb, vis_emb_dim, vis_emb_num,
    flag_layerwise_or_reweight,
    criterion_ce,criterion_ae,criterion_spc,criterion_triplet, 
    criterion_huber, criterion_infonce, criterion_kl)
    

