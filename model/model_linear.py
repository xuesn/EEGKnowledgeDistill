import torch
import torch.nn as nn

class ModelLinear(nn.Module):
    
    # 定义权值初始化----我并没用
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def __init__(self, in_dim, out_dim):
        super(ModelLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim,out_dim)
        )
        

    def forward(self, x):
        # bs * time * electrode
        # 拉成1维，接fc
        fc_in = x.reshape(x.shape[0], -1)
        return self.model(fc_in)
