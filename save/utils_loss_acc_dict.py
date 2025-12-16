


from colorama import init, Fore,  Back,  Style
init(autoreset = True)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# loss acc 记录 相关函数
def initial_loss_acc_dict():
    keys = ['loss_total', 'loss_ce', 'loss_vis_huber','loss_vis_infonce','loss_vis_kl',  'accuracy',
    
    'loss_spc', 'loss_triplet']
    values = [0,0,0,0,0,0,0,0]
    multi_loss_acc_dict=dict(zip(keys,values))
    return multi_loss_acc_dict

def tensor_loss_acc_dict(loss_acc_dict,  
                        loss_total, loss_ce, 
                        loss_vis_huber, loss_vis_infonce, loss_vis_kl,
                        right_num,
                        loss_spc,  
                        loss_triplet):
    # print(loss_total)
    loss_acc_dict['loss_total'] += loss_total.item()
    loss_acc_dict['loss_ce'] += loss_ce.item()
    loss_acc_dict['loss_vis_huber'] += loss_vis_huber.item()
    loss_acc_dict['loss_vis_infonce'] += loss_vis_infonce.item()
    loss_acc_dict['loss_vis_kl'] += loss_vis_kl.item()
    loss_acc_dict['accuracy'] += right_num
    loss_acc_dict['loss_spc'] += loss_spc.item()
    loss_acc_dict['loss_triplet'] += loss_triplet.item()
    return loss_acc_dict

def add_loss_acc_dict(dict_loss_acc_epoch,dict_loss_acc_batch):
    for key in dict_loss_acc_epoch.keys():
        dict_loss_acc_epoch[key] +=  dict_loss_acc_batch[key]
    return dict_loss_acc_epoch

def divide_loss_acc_dict(dict_loss_acc_epoch,batch_num,sample_num):
    for key in dict_loss_acc_epoch.keys():
        if 'accuracy' in key:
            dict_loss_acc_epoch[key] /= sample_num
        else:
            dict_loss_acc_epoch[key] /= batch_num
    return dict_loss_acc_epoch

def print_loss_acc_dict(loss_acc_dict, train_or_val_phase, epoch, step, batch_num,time_duration):
    if train_or_val_phase == 'train':
        fore_color = Fore.GREEN 
        # back_color = Back.GREEN 
    elif train_or_val_phase == 'val':
        fore_color = Fore.RED 
        # back_color = Back.RED 

    #打印用时
    # print('epoch:{} step:{} batch_num:{} time_duration:{}'.format(epoch,step,batch_num,time_duration))
    #打印loss
    print(fore_color + '{}:  '.format(train_or_val_phase))
    if loss_acc_dict['accuracy'] >= 1: #说明是right_num，而不是accuracy。 除非right_num=0
        print(fore_color + '    total_loss_____:{:.6f} \t accuracy:{:d}'.format(
            loss_acc_dict['loss_total'],  int(loss_acc_dict['accuracy']) ) )
    else:  # accuracy按百分比打印
        print(fore_color + '    total_loss_____:{:.6f} \t accuracy:{:.2f}% '.format(
            loss_acc_dict['loss_total'],  loss_acc_dict['accuracy']*100) )
        print('    loss_ce:{:.6f} \t loss_vh:{:.6f}'.format(
            loss_acc_dict['loss_ce'],  loss_acc_dict['loss_vis_huber']))
        print('    loss_vi:{:.6f} \t loss_vk:{:.6f}'.format(
            loss_acc_dict['loss_vis_infonce'],  loss_acc_dict['loss_vis_kl']))
        print('    loss_spc:{:.6f} \t loss_tri:{:.6f}'.format(
            loss_acc_dict['loss_spc'],  loss_acc_dict['loss_triplet']))



def dict_tolist_loss_acc(dict_loss_acc):
    list_loss_acc =[ dict_loss_acc['loss_total'],dict_loss_acc['loss_ce'],
    dict_loss_acc['loss_vis_huber'],dict_loss_acc['loss_vis_infonce'],dict_loss_acc['loss_vis_kl'],dict_loss_acc['accuracy'],
    dict_loss_acc['loss_spc'],dict_loss_acc['loss_triplet'],
    ]
    return list_loss_acc


