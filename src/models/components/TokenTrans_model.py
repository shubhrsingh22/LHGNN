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

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()



class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    
class PatchEmbed(nn.Module):

    def __init__(self,in_chans=1,out_chans=768):

        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans,out_chans//2,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.GELU(),
            nn.BatchNorm2d(out_chans//2),
            nn.Conv2d(out_chans//2,out_chans//2,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.GELU(),
            nn.BatchNorm2d(out_chans//2),
            nn.Conv2d(out_chans//2,out_chans,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.GELU(),
            nn.BatchNorm2d(out_chans),
            nn.Conv2d(out_chans,out_chans,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.GELU(),
            nn.BatchNorm2d(out_chans),
        )

    def forward(self,x):
        x = self.proj(x)
        return x
class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class StokenAttention(nn.Module):

    def __init__(self,dim,n_iter=1,stoken_size=None,refine=True,refine_attention=True,num_heads=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        
        super().__init__()
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.fold = Fold(3)
        self.unfold = Unfold(3)

        if refine:
            self.stoken_refine = Attention(dim=dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop, proj_drop=proj_drop)


    def stoken_forward(self,x):

        B,C,H0,W0 = x.shape
        
        h,w = self.stoken_size
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x,(pad_l,pad_r,pad_t,pad_b))
        _,_,H,W = x.shape
        hh,ww = H//h,W//w
        stoken_features = F.adaptive_avg_pool2d(x,(hh,ww))
        pixel_features = x.reshape(B,C,hh,h,w,ww).permute(0,2,4,3,5,1).reshape(B,hh*ww,h*w,C)

        with torch.no_grad():

            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)
                stoken_features = stoken_features.transpose(1,2).reshape(B,hh*ww,C,9)
                affinity_matrix = pixel_features @ stoken_features * self.scale
                affinity_matrix = affinity_matrix.softmax(dim=-1)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter -1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12)
                

        stoken_features = pixel_features.transpose(-1,-2) @ affinity_matrix
        stoken_features = self.fold(stoken_features.permute(0,2,3,1).reshape(B*C,9,hh,ww)).reshape(B,C,hh,ww)
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12)
        
        if self.refine:
            if self.refine_attention:
                stoken_features = self.stoken_refine(stoken_features)
        

        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        # 714
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        # 687
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # 681
        # 591 for 2 iters
                
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    def forward(self,x):
        x = self.stoken_forward(x)
        return x
    
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        drop_path = 0.1
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer()
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
    

class StokenAttentionLayer(nn.Module):

    def __init__(self,dim,n_iter,stoken_size,num_heads=1,mlp_ratio=4.,qkv_bias=False,qk_scale=None,drop=0.,
                 attn_drop=0.,drop_path=0.,act_layer=nn.GELU,layerscale=False,init_values=1e-6):
        super().__init__()

        self.layerscale = layerscale
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim,n_iter,stoken_size,num_heads=num_heads,qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim,eps=1e-6)
        self.mlp = Mlp(in_features=dim,hidden_features=int(dim*mlp_ratio),out_features=dim,act_layer=act_layer,drop=drop,downsample=False)
    
    def forward(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):

    def __init__(self,num_layers,dim,n_iter,stoken_size,num_heads=1,mlp_ratio=4.0,qkv_bias=False,qk_scale=False,drop_rate=0.,attn_drop=0.,
                 drop_path=0.,act_layer=nn.GELU,layerscale=False,init_values=1e-6,downsample=False):
        
        super().__init__()
        self.blocks = nn.ModuleList([])
        
        self.blocks = nn.ModuleList([StokenAttentionLayer(dim[0],n_iter=n_iter,stoken_size=stoken_size,num_heads=num_heads,mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop,drop_path=drop_path[i] if isinstance(drop_path,list)else drop_path,
                                                           act_layer=act_layer,layerscale=layerscale,init_values=init_values) for i in range(num_layers)])
        
        
        if downsample:
            self.downsample = PatchMerging(dim[0],dim[1])
        else:
            self.downsample = None
    
    def forward(self,x):

        for idx,blk in enumerate(self.blocks):
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)  
        return x
            
class TokenTransformer(nn.Module):

    def __init__(self,num_classes=200,size='s',freq_num=128,time_num=1024,n_iter=[1,1,1,1],stoken_size=[8,4,2,1],mlp_ratio=4.0,qkv_bias
               =True,qk_scale=None,drop_rate=0.0,attn_drop_rate=0.0,drop_path=0.1,projection=None,init_values=1e-6,layerscale=[False,False,False,False]):
        super().__init__()
        self.num_classes = num_classes
        self.size = size
        self.swish = MemoryEfficientSwish()
       
        if size == 's':
            self.blocks = [3, 5, 9, 3]
            self.channels = [64,128,320,512]
            self.num_heads = [1,2,5,8]
            self.n_iter =  [1,1,1,1]
            self.stoken_size = [8,4,1,1]
            self.projection = 1024
            self.mlp_ratio = 4.0
            self.qkv_bias = True
            self.qkv_scale = None
            self.drop_rate = drop_rate
            self.drop_path = drop_path
            self.layerscale = [False,False,False,False]
            init_values = 1e-6
        
        elif size == 'm':
            self.blocks = [4,6,14,6]
            self.channels = [96,192,384,512]
            self.num_heads = [2,3,6,8]
            self.n_iter = [1,1,1,1]
            self.stoken_size = [8,4,1,1]
            self.projection = 1024
            self.mlp_ratio = 4.0
            self.qkv_bias = True
            self.qkv_scale = None
            self.drop_rate = drop_rate
            self.drop_path = drop_path
            self.layerscale = [False,False,False,False]
            init_values = 1e-6
        
        elif size == 'l':

            self.blocks = [4, 7, 19, 8]
            self.channels = [96, 192, 448, 640]
            self.num_heads = [2, 3, 7, 10]
            self.n_iter = [1,1,1,1]
            self.stoken_size = [8,4,1,1]
            self.projection = 1024
            self.mlp_ratio = 4.0
            self.qkv_bias = True
            self.qkv_scale = None
            self.drop_rate = drop_rate
            self.drop_path = drop_path
            self.layerscale = [False,False,False,False]
            init_values = 1e-6


        

        
        self.num_layers = len(self.blocks)
        self.num_features = self.channels[-1]
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(1,self.channels[0])
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.blocks))]

        self.layers = nn.ModuleList([])

        for i_layer in range(self.num_layers):

            layer = BasicLayer(num_layers=self.blocks[i_layer],
                               dim=[self.channels[i_layer], self.channels[i_layer+1] if i_layer<self.num_layers-1 else None],n_iter=n_iter[i_layer],stoken_size=to_2tuple(stoken_size[i_layer]),
                               num_heads = self.num_heads[i_layer],mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias,qk_scale=self.qkv_scale,
                               drop_rate=self.drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(self.blocks[:i_layer]):sum(self.blocks[:i_layer+1])],
                               downsample=i_layer<self.num_layers-1,
                               layerscale=self.layerscale[i_layer],
                               init_values=init_values)
            self.layers.append(layer)
        
        self.proj = nn.Conv2d(self.num_features,self.projection,1) 
        self.norm = nn.BatchNorm2d(self.projection)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.projection,num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    

    def forward_features(self,x):
        x = self.patch_embed(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.proj(x)
        x = self.norm(x)
        x = self.swish(x)

        x = self.avgpool(x).flatten(1)
        return x
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2,3)
        x = self.forward_features(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x








        