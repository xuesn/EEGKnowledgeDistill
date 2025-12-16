
'''https://github.com/MiaoZhengQing/LMDA-Code'''


import torch
import torch.nn as nn



def square_activation(x):
    return torch.square(x)

def safe_log(x):
    return torch.clip(torch.log(x), min=1e-7, max=1e7)

class ShallowConvNet(nn.Module):
    # def __init__(self, num_classes, chans, samples=1125):
    def __init__(self,  chans, samples=1125):
        # chans 即electrode
        # samples 即timepoint_num
        super(ShallowConvNet, self).__init__()
        self.conv_nums = 64
        self.features = nn.Sequential(
            nn.Conv2d(1, self.conv_nums, (1, 3)),
            nn.Conv2d(self.conv_nums, self.conv_nums, (chans, 1), bias=False),
            nn.BatchNorm2d(self.conv_nums)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 1))
        # self.dropout = nn.Dropout()
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        out = self.avgpool(out)
        n_out_time = out.cpu().data.numpy().shape
        # self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)
    def forward(self, x):
        # input: bs*time*ch
        x = x.transpose(2,1)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = square_activation(x)
        x = self.avgpool(x)
        x = safe_log(x)
        # x = self.dropout(x)
        features = torch.flatten(x, 1)  # 使用卷积网络代替全连接层进行分类, 因此需要返回x和卷积层个数
        # cls = self.classifier(features)
        # return cls
        return features
