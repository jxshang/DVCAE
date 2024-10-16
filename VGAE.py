#导入相应的包
from dgl.nn.pytorch import GraphConv,GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F


#定义VGAEModel
class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, z_dim, device):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        #三层GraphConv，原文中生成均值和方差的W0是共享的，W1是不同的，因此一共要三层
        #https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/conv/graphconv.html
        #GraphConv用于实现GCN的卷积
        layers = [GATConv(self.in_dim, int(self.hidden_dim / 8), num_heads = 8, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=F.relu, allow_zero_in_degree=True),  # 第一层，共享参数
                  GATConv(self.hidden_dim, int(self.z_dim / 8), num_heads = 8, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=lambda x: x, allow_zero_in_degree=True),  # 第二层求均值
                  GATConv(self.hidden_dim, int(self.z_dim / 8), num_heads = 8, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=lambda x: x, allow_zero_in_degree=True)]  # 第二层求方差
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        h = self.layers[0](g, features)#第一层得到输出h
        h = h.view(-1, self.hidden_dim)
        self.mean = self.layers[1](g, h).view(-1, self.z_dim)#第二层求均值
        self.log_std = self.layers[2](g, h).view(-1, self.z_dim)#第二层求方差
        gaussian_noise = torch.randn(features.size(0), self.z_dim).to(torch.float32).to(self.device)#标准高斯分布采样，大小是features_size*hidden2_dim
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)#这里其实是reparameterization trick，具体看公式1和代码如何对应
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))#解码器点乘还原邻接矩阵A'
        return adj_rec

    def forward(self, g):#前向传播
        features = g.ndata['x']
        z = self.encoder(g, features)#编码器得到隐变量
        adj_rec = self.decoder(z)#解码器还原邻接矩阵
        return z, adj_rec
