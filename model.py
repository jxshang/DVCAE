import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from VGAE import VGAEModel
from VTAE import VTAEModel

class CTGVAEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim1,  z_dim, out_dim, max_seq, device, hidden_dim2 = 128, hidden_dim3 = 64):
        super(CTGVAEModel, self).__init__()
        # 一个VGAE模型。
        self.vgae_model = VGAEModel(in_dim, hidden_dim1, z_dim, device)
        self.vgte_model = VTAEModel(in_dim, hidden_dim1, z_dim, max_seq, device)
        # 输出层
        self.out_layers = nn.ModuleList([
            nn.Linear(z_dim * 2, hidden_dim2), #mlp1
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3), # mlp2
            nn.ReLU(),
            nn.Linear(hidden_dim3, out_dim) #output
        ])


    def forward(self, g, x):
        # VGAE
        z_temp, adj_rec = self.vgae_model(g) # z_temp (batch, nodes, z_dim) adj_rec (batch, nodes, nodes)
        g.ndata['z'] = z_temp
        z_vgae = dgl.mean_nodes(g, 'z') # z_vgae (batch, z_dim)
        # VTAE
        z_vtae, x_rec = self.vgte_model(x) # z_vtae (batch, z_dim) x_rec (batch, max_seq, in_dim)

        z = torch.cat((z_vtae, z_vgae), 1) # z (batch, z_dim * 2)

        # output
        pre = z
        for conv in self.out_layers:
            pre = conv(pre) # pre (batch, 1)
        return pre, z, adj_rec, x_rec


