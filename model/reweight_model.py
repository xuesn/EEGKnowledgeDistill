import torch
import torch.nn as nn

class ModelLinearReweight(nn.Module):
    
    # 定义权值初始化----没使用
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
        super(ModelLinearReweight, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim,out_dim)
        )
        

    def forward(self, x):
        # bs * vis_emb_num * vis_emb_dim
        # print(x.shape)
        transposed_x = torch.transpose(x, dim0=1, dim1=2)
        # print(transposed_x.shape)
        reweighted_x = self.model(transposed_x)
        # print(reweighted_x.shape)
        transposed_reweighted_x = torch.transpose(reweighted_x, dim0=1, dim1=2)
        # print(transposed_reweighted_x.shape)
        return transposed_reweighted_x

        

