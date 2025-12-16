'''https://github.com/MiaoZhengQing/LMDA-Code'''


import torch
import torch.nn as nn


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel1_size,  **kw):
        super(SeparableConv2D, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel1_size, **kw),
            # nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            # pw
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), **kw),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        return self.depth_conv(x)

class EEGNet(nn.Module):
    # def __init__(self, num_classes, chans, samples=1125, dropout_rate=0.5, kernel_length=64, F1=8,
    #              F2=16,):
    def __init__(self,  chans, samples=1125, dropout_rate=0.5, kernel_length=64, F1=8,
                 F2=16,):
        # F1 F2 为conv的channel即特征数
        # chans 即electrode
        # samples 即timepoint_num
        super(EEGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, kernel_size=(chans, 1), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            # nn.ReLU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate),
            # for SeparableCon2D
            # SeparableConv2D(F1, F2, kernel1_size=(1, 16), bias=False),
            nn.Conv2d(F1, F1, kernel_size=(1, 3), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            # nn.ReLU(),
            nn.Conv2d(F1, F2, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(F2),
            # nn.ReLU(),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(p=dropout_rate),
        )
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        n_out_time = out.cpu().data.numpy().shape
        # self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)
    def forward(self, x):
        # input: bs*time*ch
        x = x.transpose(2,1)
        x = x.unsqueeze(1)
        conv_features = self.features(x)
        features = torch.flatten(conv_features, 1)
        # cls = self.classifier(features)
        # return cls
        return features
    

