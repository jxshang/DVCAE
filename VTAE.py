import torch
import torch.nn as nn
import torch.nn.functional as F


class VTAEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, z_dim, max_seq, device):
        super(VTAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.max_seq = max_seq
        self.device = device

        self.en_gru_layer = nn.GRU(input_size = self.in_dim, hidden_size = self.hidden_dim, num_layers=1, bidirectional=True, dropout=0.5)  # 第一层，通过双向的LSTM将cascade进行处理,输出
        self.en_dense_layer1 = nn.Linear(self.hidden_dim * 2, self.z_dim)  # 均值 向量
        self.en_dense_layer2 = nn.Linear(self.hidden_dim * 2, self.z_dim)  # 保准方差 向量

        self.de_gru_layer = nn.GRU(input_size = self.z_dim, hidden_size = self.hidden_dim, num_layers=1, bidirectional=True, dropout=0.5)  # 解码层，输入hidden2-dim 输出 input_dim
        self.de_dense_layer = nn.Linear(self.hidden_dim * 2, self.in_dim)

    def encode(self, x):
        z, h_n = self.en_gru_layer(x) # z (batch, seq, hidden_dim * 2)
        # 获取反向的最后一个output 和 正向的最后一个output
        z_last = F.relu(torch.cat((z[:, -1, :self.hidden_dim], z[:, 0, -self.hidden_dim:]), 1)) # z_last (batch, hidden_dim + hidden_dim)
        self.mean = self.en_dense_layer1(z_last) # mean (batch, z_dim)
        self.log_std = self.en_dense_layer2(z_last) #
        self.std = torch.exp(self.log_std)
        gaussian_noise = torch.randn(x.size(0), self.z_dim).to(torch.float32).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device) # (batch, hidden2_dim)

        return sampled_z

    def decode(self, z): #(batch, z_dim)
        z = torch.repeat_interleave(z.unsqueeze(dim=1), repeats=self.max_seq, dim=1) # z (batch, max_seq, hidden2_dim)
        x, h_n = self.de_gru_layer(z) # x (batch, max_seq, hidden_dim * 2)
        x = F.relu(x)
        x = self.de_dense_layer(x) # x (batch, max_seq, input_dim)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec
