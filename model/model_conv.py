import torch
from torch import nn



# 复现“Learning spatiotemporal graph representations for visual perception using eeg signals,” 中的时间卷积这一支


class ModelConv(nn.Module):


    # 定义权值初始化----未使用
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # kernal1-3是时间维度1维卷积；kernal4是所有电极加权
    def __init__(self,  ch1=16, ch2=32, ch3=64, ch4=64,
                 kernal1=5, kernal2=5, kernal3=5, kernal4=124,
                 pool_stride=2,
                 dropout_p=0.25,
                 activate_type='elu',
                 pool_type='max',
                 electrode_num=124,  
                 ):  # 换新数据集了 class_num=6类或72张
        super(ModelConv, self).__init__()

        # 选择激活函数--------------------------------------------------
        if activate_type == 'relu':
            self.activate_func = nn.ReLU()
        elif activate_type == 'leaky-relu':
            self.activate_func = nn.LeakyReLU(negative_slope=0.1)
        elif activate_type == 'elu':
            self.activate_func = nn.ELU(alpha=1.0)
        elif activate_type == 'gelu':
            self.activate_func = nn.GELU()
        else:
            print('!!Undefined Activate Function!!')
            self.activate_func = nn.ELU()

        # 选择pooling方式--------------------------------------------------
        if pool_type == 'max':
            self.pool_layer = nn.MaxPool2d(kernel_size=(pool_stride, 1),
                                           stride=(pool_stride, 1))
        elif pool_type == 'mean':
            self.pool_layer = nn.AvgPool2d(kernel_size=(pool_stride, 1),
                                           stride=(pool_stride, 1))
        else:
            print('!!Undefined Pooling Type!!')
            self.pool_layer = nn.MaxPool2d(kernel_size=(pool_stride, 1),
                                           stride=(pool_stride, 1))

        # 1*1卷积层--------------------------------------------------
        self.subject_layer = nn.Sequential(
            # 简单整合所有电极信息，forward中再调整数据维度顺序
            nn.Conv2d(in_channels=1, out_channels=electrode_num,
                      kernel_size=(1, electrode_num), stride=(1, 1)),
        )

        # 4层时间卷积模型--------------------------------------------------
        # input_data: batch,32, 124
        self.conv_4layers = nn.Sequential(
            # layer1
            nn.Conv2d(in_channels=1, out_channels=ch1,
                      kernel_size=(kernal1, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=ch1),
            self.activate_func,
            # layer2
            nn.Conv2d(in_channels=ch1, out_channels=ch2,
                      kernel_size=(kernal2, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=ch2),
            # 作者没说激活函数放在哪里，我就按照网上推荐conv -> bn->relu -> pooling
            self.activate_func,
            # self.pool_layer1,
            self.pool_layer,
            # layer3
            nn.Dropout(p=dropout_p),
            nn.Conv2d(in_channels=ch2, out_channels=ch3,
                      kernel_size=(kernal3, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=ch3),
            self.activate_func,
            # self.pool_layer2,
            self.pool_layer,
            # layer4
            nn.Dropout(p=dropout_p),
            nn.Conv2d(in_channels=ch3, out_channels=ch4,
                      kernel_size=(1, kernal4), stride=(1, 1)),
            nn.BatchNorm2d(num_features=ch4),
            self.activate_func,
        )

    def forward(self, in_data,):
        # in_data: batch,32, 124
        in_data_float = in_data.to(torch.float32)
        in_data_unsqueeze = in_data_float.unsqueeze(dim=1)
        # batch 1 32 124

        # subject-layer-adapter
        sub_adapt_data = self.subject_layer(in_data_unsqueeze)
        # batch ch time electrode
        # batch 124 32 1
        # torch.transpose (input, dim0, dim1) 函数将输入张量 input 的第 dim0 个维度和第 dim1 个维度进行交换，并将交换维度后的张量返回。
        sub_adapt_data = torch.transpose(sub_adapt_data, dim0=1, dim1=3)

        # conv-encode
        conv_out = self.conv_4layers(sub_adapt_data)

        return conv_out
