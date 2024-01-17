import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
 
#from stgcn_traffic_prediction.pygcn.layers import GraphConvolution
from .period import period
from .closeness import close
from .spatial import Spatial
from .utils import getadj,getA_cosin,getA_corr

class Fusion(nn.Module):
    def __init__(self,dim_in):
        super(Fusion,self).__init__()
        self.weight2 = nn.Linear(dim_in*2,dim_in)

    def forward(self,x1,x2=None):
        if(x2 is not None):
            out = self.weight2(torch.cat([x1,x2],dim=-1))
        else:
            out = x1
        return out

class T_STGCN(nn.Module):
    def __init__(self,len_closeness, external_size, N, k, spatial, s_model_d,c_model_d,p_model_d,t_model_d,dim_hid=16, drop_rate=0.1):
        super(T_STGCN,self).__init__()

        self.spatial = Spatial(len_closeness,k,N,s_model_d)#输出400*3
        self.c_temporal = close(k,N,c_model_d)
        self.spatial_f = Spatial(len_closeness,k,N,s_model_d)
        self.fusion = Fusion(len_closeness)
        self.k = k
    def forward(self,x_c,mode,c,s,FS):
        bs = len(x_c)
        N = x_c.shape[-1]
        len_closeness = x_c.shape[1]
        x_spatial = None
        sq_c = None
        #print('x_c\n',x_c)
        #print(x_c.shape)
        adj = getA_corr(x_c.permute((0,2,3,1)))
        index = torch.argsort(adj,dim=-1,descending=True)[:,:,0:self.k]
        
        if(s):
            #spatial
            x_spatial,_ = self.spatial(x_c,None,mode,adj,index)
            #print('spatial:',x_spatial[0])
            #x_c,x_p,tgt_mode,mode,flow,A=None,index=None,x_t=None
        #temporal
        if(c):
            sq_c = F.sigmoid(self.c_temporal(x_c,mode,adj,index))
            #print('sq_c:',sq_c[0])

        
        # print('x_c:',x_c.shape)#[2, 3, 1, 4000]
        # print('sq_c:',sq_c.shape)#([2, 4000, 3]
        # print('sq_c.transpose(1,2).unsqueeze(-2).unsqueeze(1)',sq_c.transpose(1,2).unsqueeze(-2).unsqueeze(1).shape)#torch.Size([2, 1, 3, 1, 4000])
        x_temporal=sq_c
        if(FS):
            x_temporal,_ = self.spatial_f(x_c,sq_c.transpose(1,2).unsqueeze(-2).unsqueeze(1),mode,adj,index)
            
        #fusion
        pred = self.fusion(x_temporal,x_spatial)
        return pred.transpose(1,2)


