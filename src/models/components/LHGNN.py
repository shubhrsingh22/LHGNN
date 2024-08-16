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
from src.utilities.utilities.model_utils import FFN,ResDWC,ConvFFN,Stem_conv,DownSample

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



class LHGNN(nn.Module):

    def __init__(self,k=9,act='gelu',norm='batch',bias=True,dropout=0.0,dilation=True,epsilon=0.2,drop_path=0.3,size ='s',
               num_class=200,emb_dims=1024,freq_num=128,time_num=1024,clusters=50,cluster_ratio=0.5,conv='mr'):
        
        super(LHGNN,self).__init__()
        
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
       
        self.bias = bias
              
        self.drop_path = drop_path
        
        self.num_class = num_class
        self.emb_dims = emb_dims
        self.freq_num = freq_num
        self.time_num = time_num
        self.epsilon = epsilon
        self.dilation = dilation
        self.dropout = dropout
        stochastic = False
        self.num_blocks = sum(self.blocks)
        
        self.cluster_ratio = cluster_ratio
        if conv == 'lhg':
            channel_mul = 3
        else:
            channel_mul = 2
            
        self.conv = conv
        reduce_ratios = [4,2,1,1]
        num_clusters = [int (x.item()) for x in torch.linspace(clusters,clusters,self.num_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k,k,self.num_blocks)]
        max_dilation = 128//max(num_clusters)
       
        self.stem = Stem_conv(1,self.channels[0],act=act)
        
        
        self.pos_embed = nn.Parameter(torch.zeros(1,self.channels[0],freq_num//4,time_num//4))

        #Num nodes after the stem block 
        self.HW = freq_num//4 * time_num//4 
        
        #Drop path 
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(DownSample(self.channels[i-1], self.channels[i]))
                self.HW = self.HW // 4
            for j in range(self.blocks[i]):
                self.backbone += [
                    Seq(Grapher(self.channels[i], num_knn[idx],num_clusters[idx], min(idx // 4 + 1, max_dilation), self.conv, self.act, self.norm,
                                    self.bias, stochastic, epsilon, reduce_ratios[i], n=self.HW, drop_path=dpr[idx],
                                    relative_pos=True,cluster_ratio=self.cluster_ratio,channel_mul=channel_mul),
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
            
        preds = torch.sigmoid(x)
        
        preds = preds.squeeze(-1).squeeze(-1)
        
            
        
        return preds