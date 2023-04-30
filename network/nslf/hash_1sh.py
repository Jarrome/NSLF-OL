import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from .encoding import get_encoder
from .sh import eval_sh



class Hash1SH(nn.Module):
    def __init__(self,
                resolution = [256]*3,
                #resolution = [64*4]*3,
                color_rank=[1]*3,
                degree=3,
                bound=1):
        super().__init__()

        self.degree = degree
        #self.enc_dir_dim = 3


        self.encoder, self.in_dim = get_encoder('hashgrid', desired_resolution=512 * bound)#, num_levels=16, level_dim=2)# *6 * 13)

        net_width= 32
        self.n_sphere = 1
        self.late_mlp = nn.Sequential(
            nn.Linear(self.in_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, (degree+1)**2*self.n_sphere),
            )

        self.weight_encoder, self.in_dim = get_encoder('hashgrid', desired_resolution=512 * bound)#, num_levels=16, level_dim=2)# *6 * 13)
        self.weight_mlp = nn.Sequential(
            nn.Linear(self.in_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 3*3*2+3*2 + self.n_sphere*3+3),
            )



        def init_param(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.late_mlp.apply(init_param)
        self.weight_mlp.apply(init_param)


        self.encode_param = [*(self.encoder.parameters()),*(self.weight_encoder.parameters())]
        self.decode_param = [*(self.late_mlp.parameters()),*(self.weight_mlp.parameters())]


 



    def forward(self, xs, ds):
        '''
            [n,3],[n,3] in [-.5,.5],[-1,1]
            xs is the point, ds is the direction
        '''
        n = xs.shape[0]
        xs_ = xs * 2
        xs_encoded = self.encoder(xs_) #N, 32

        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        #xs_encoded = torch.concat([xs_encoded,ds],axis=-1)
        shs = self.late_mlp(xs_encoded) # N,(degree+1)**2
        c_is = eval_sh(self.degree, shs.view(-1,self.n_sphere,(self.degree+1)**2), ds) # n,s


        # weight
        xs_encoded = self.weight_encoder(xs_)
        ws = self.weight_mlp(xs_encoded) # n,3*3+3+3+3

        l1_ed = self.n_sphere*3+3
        layer1 = ws[:,:l1_ed] # n,3s+3
        layer2 = ws[:,l1_ed:l1_ed+3*3+3]
        layer3 = ws[:,l1_ed+3*3+3:]

        
        #c_is = c_is * layer1[:,:-3] + layer1[:,-3:] # n,3
        c_is = torch.bmm(c_is.unsqueeze(1), layer1[:,:l1_ed-3].view(n, self.n_sphere, 3)).squeeze() + layer1[:,-3:]
        c_is = nn.functional.relu(c_is)
        c_is = torch.bmm(c_is.unsqueeze(1), layer2[:,:9].view(n,3,3)).squeeze() + layer2[:,9:]
        c_is = nn.functional.relu(c_is)
        c_is = torch.bmm(c_is.unsqueeze(1), layer3[:,:9].view(n,3,3)).squeeze() + layer3[:,9:]
        c_is = nn.functional.sigmoid(c_is)

        return {"c_is": c_is, "sigma_is": None}#sigma_is.float()}

