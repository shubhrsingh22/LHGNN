import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision 
import torchvision.models as models
from torch.nn import MultiheadAttention
from torch.nn import Sequential as Seq
from timm.models.layers import to_2tuple,trunc_normal_
from timm.models.layers import DropPath
from src.models.gcn_lib1.torch_nn import act_layer, norm_layer, MLP, BasicConv

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu',drop_path=0.0):
        super(FFN,self).__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer(act)
        self.fc1 = Seq(nn.Conv2d(in_features,hidden_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(hidden_features))
        self.fc2 = Seq(nn.Conv2d(hidden_features,out_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(out_features))
        
                          
       

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
       
        return x

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ResDWC,self).__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, 1,bias=True, groups=dim)
                
        # self.conv_constant = nn.Parameter(torch.eye(kernel_size).reshape(dim, 1, kernel_size, kernel_size))
        # self.conv_constant.requires_grad = False
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x) 
        
        # return F.conv2d(x, self.conv.weight+self.conv_constant, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
        return x

class ConvFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu',drop_path=0.0):

        super(ConvFFN,self).__init__()

        self.out_features = out_features if out_features is not None else in_features
        self.hidden_features = hidden_features if hidden_features is not None else in_features
        self.fc1 = Seq(nn.Conv2d(in_features,hidden_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(hidden_features))
        self.act = act_layer(act)
        self.fc2 = Seq(nn.Conv2d(hidden_features,out_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(out_features))
        self.conv = ResDWC(hidden_features, 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        #x,hyperedges = inputs
        B, C, H, W = x.shape
        #x = x.reshape(B, C, -1, 1).contiguous()
        shortcut = x
        x = self.fc1(x)
        
        x = self.act(x)
        
        x = self.conv(x)
        
        x = self.fc2(x)
        
        x = self.drop_path(x) + shortcut
        return x


class Stem_conv(nn.Module):

    def __init__(self,in_dim=1,out_dim=None,act='gelu'):
        super(Stem_conv,self).__init__()

        self.convs = Seq(nn.Conv2d(in_dim,out_dim//2,3,stride=2,padding=1),
                        nn.BatchNorm2d(out_dim//2),
                        act_layer(act),
                        nn.Conv2d(out_dim//2,out_dim,3,stride=2,padding=1),
                        nn.BatchNorm2d(out_dim),
                        act_layer(act),
                        nn.Conv2d(out_dim,out_dim,3,stride=1,padding=1),
                        nn.BatchNorm2d(out_dim),
                         )
    
    def forward(self,x):
        x = self.convs(x)
        return x

class DownSample(nn.Module):

    def __init__(self,in_dim,out_dim=768,act='relu'):
        super(DownSample,self).__init__()
        self.conv = Seq(nn.Conv2d(in_dim,out_dim,3,stride=2,padding=1),
                        nn.BatchNorm2d(out_dim))
    
    def forward(self,x):

        x = self.conv(x) 
        return x