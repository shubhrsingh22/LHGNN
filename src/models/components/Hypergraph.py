import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath
from src.models.gcn_lib1.torch_vertex import Grapher
from src.models.gcn_lib1.torch_nn import act_layer, norm_layer, MLP, BasicConv
from timm.models.layers import to_2tuple,trunc_normal_
from timm.models.layers import DropPath
import timm
import torchvision 
import torchvision.models as models
from torch.nn import MultiheadAttention

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)


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

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super(GroupWiseLinear,self).__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Transformer(nn.Module):

    def __init__(self, dim, depth=1, heads=1,dropout=0.1,act='gelu'):

        super(Transformer,self).__init__()
        self.multihead_attn = MultiheadAttention(dim, heads, dropout=dropout,batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.query_ffn = FFN(dim, dim*4,dim, act=act, drop_path=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_label, key_x, value_x):
        
        out,attn_matrix = self.multihead_attn(query_label, key_x, value_x)
       
        out = query_label + self.dropout(out)

        out = self.norm1(out)
        out = self.query_ffn(out.transpose(1,2).unsqueeze(-1))
        out = self.norm2(out.transpose(1,2).squeeze(-1)) # B,200,1024
       
        return out

class HGCN(nn.Module):

    def __init__(self,k=9,act='gelu',norm='batch',bias=True,dropout=0.0,dilation=True,epsilon=0.2,drop_path=0.3,size ='s',
               num_class=200,emb_dims=1024,freq_num=128,time_num=1024):
        
        super(HGCN,self).__init__()
        
        if size == 's':
            self.blocks = [2, 2, 6, 2]
            
            #self.channels = [64, 128, 320, 512]
            self.channels = [80, 160, 400, 640]
            self.emb_dims = 1024
        elif size == 'm':
            self.blocks = [2,2,16,2]
            self.channels = [96, 192, 384, 768]
            self.emb_dims = 1024
        else:
            self.blocks = [2,2,18,2]
            self.channels = [128, 256, 512, 1024]
        self.k = int(k)  # number of edges per node  
        self.act = act      
        
        self.norm = norm
        print(f'norm is {self.norm}')
        self.bias = bias
        print(f'bias is {self.bias}')        
        self.drop_path = drop_path
        print(f'drop_path is {self.drop_path}')
        self.num_class = num_class
        self.emb_dims = emb_dims
        self.freq_num = freq_num
        self.time_num = time_num
        self.epsilon = epsilon
        self.dilation = dilation
        self.dropout = dropout
        stochastic = False
        self.num_blocks = sum(self.blocks)
        
        
        
# Remove or replace the fully connected layer
        
        self.proj = nn.Conv2d(512, self.channels[0], kernel_size=1, stride=1, padding=0)
        self.conv = 'mr'
        reduce_ratios = [4,2,1,1]
        num_clusters = [int(x.item()) for x in torch.linspace(k,k,self.num_blocks)]
        graph_params = num_clusters
        num_centroids = [int(x.item()) for x in torch.linspace(50,50,self.num_blocks)]
        max_dilation = 128//max(num_clusters)
       
        self.stem = Stem_conv(1,self.channels[0],act=act)
        self.query_embed = nn.Embedding(num_class, self.channels[-1])
        self.label_proj = nn.Conv2d(self.channels[-1],1024, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1,self.channels[0],freq_num//4,time_num//4))
        self.HW = freq_num//4 * time_num//4 

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(DownSample(self.channels[i-1], self.channels[i]))
                self.HW = self.HW // 4
            for j in range(self.blocks[i]):
                self.backbone += [
                    Seq(Grapher(self.channels[i], graph_params[idx], min(idx // 4 + 1, max_dilation), self.conv, self.act, self.norm,
                                    self.bias, stochastic, epsilon, reduce_ratios[i], n=self.HW, drop_path=dpr[idx],
                                    relative_pos=True,num_centroids=num_centroids[idx]),
                          ConvFFN(in_features=self.channels[i],hidden_features= self.channels[i] * 4,out_features=self.channels[i], act=act, drop_path=dpr[idx])
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)
        
        self.prediction = Seq(nn.Conv2d(self.channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(self.dropout),
                              nn.Conv2d(1024, self.num_class, 1, bias=True))
        
                              
        
        self.model_init()
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
                
    def forward(self,inputs):
        
        inputs = inputs.unsqueeze(1)
        inputs = inputs.transpose(2,3)
        x = self.stem(inputs) + self.pos_embed
        

        
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            
            x = self.backbone[i](x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        
        x = self.prediction(x)
            
        #preds = torch.sigmoid(x)
        
        preds = preds.squeeze(-1).squeeze(-1)
        
            
        
        return preds