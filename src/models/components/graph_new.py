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

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu',drop_path=0.0):
        super().__init__()
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


class Stem_conv(nn.Module):

    def __init__(self,in_dim=1,out_dim=None,act='gelu'):
        super().__init__()

        self.conv = Seq(nn.Conv2d(in_dim,out_dim//2,3,stride=2,padding=1),
                        nn.BatchNorm2d(out_dim//2),
                        act_layer(act),
                        nn.Conv2d(out_dim//2,out_dim,3,stride=2,padding=1),
                        nn.BatchNorm2d(out_dim),
                        act_layer(act),
                        nn.Conv2d(out_dim,out_dim,3,stride=1,padding=1),
                        nn.BatchNorm2d(out_dim),
                         )
    
    def forward(self,x):
        x = self.conv(x)
        return x

class DownSample(nn.Module):

    def __init__(self,in_dim,out_dim=768,act='relu'):
        super().__init__()
        self.conv = Seq(nn.Conv2d(in_dim,out_dim,3,stride=2,padding=1),
                        nn.BatchNorm2d(out_dim))
    
    def forward(self,x):

        x = self.conv(x)
        return x

class HGCN(nn.Module):

    def __init__(self,k=9,act='gelu',norm='batch',bias=True,dropout=0.0,dilation=True,epsilon=0.2,drop_path=0.1,size ='m',
               num_class=200,emb_dims=1024,freq_num=128,time_num=1024):
        
        super().__init__()
        
        if size == 's':
            self.blocks = [2, 2, 6, 2]
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
        
        self.conv = 'mr'
        reduce_ratios = [4,2,1,1]
        num_clusters = [int(x.item()) for x in torch.linspace(k,k,self.num_blocks)]
        graph_params = num_clusters
        max_dilation = 128//max(num_clusters)
       
        self.stem = Stem_conv(1,self.channels[0],act=act)

        self.pos_emb = nn.Parameter(torch.zeros(1,self.channels[0],freq_num//4,time_num//4))

        HW = freq_num//4 * time_num//4
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(DownSample(self.channels[i-1], self.channels[i]))
                HW = HW // 4
            for j in range(self.blocks[i]):
                self.backbone += [
                    Seq(Grapher(self.channels[i], graph_params[idx], min(idx // 4 + 1, max_dilation), self.conv, self.act, self.norm,
                                    self.bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(self.channels[i], self.channels[i] * 4, act=act, drop_path=dpr[idx])
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
        x = self.stem(inputs) + self.pos_emb
        
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        
        x = self.prediction(x)
        
        preds = torch.sigmoid(x)
        
        preds = preds.squeeze(-1).squeeze(-1)
        
        logits = x.squeeze(-1).squeeze(-1)
        
        return logits,preds

class PatchEmbed_ast(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Vig_pyr(nn.Module):
    def __init__(self,k=9,act='gelu',norm='batch',bias=True,dropout=0.0,dilation=True,epsilon=0.2,drop_path=0.1,size ='m',
               num_class=200,emb_dims=1024,freq_num=128,time_num=1024):
        super().__init__()

        #k = int(k)  # number of edges per node
        self.k = k
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
        model_size = 's'

        if model_size =='t':
            blocks = [2,2,6,2]
            channels = [48, 96, 240, 384]

        elif model_size == 's' :
            print("Using small model")
            blocks = [2,2,6,2]
            channels = [80, 160, 400, 640]
            plg_blocks = [1,1,3,1]
        elif model_size == 'm':
            print("Using Medium model")
            blocks = [2,2,16,2]
            plg_blocks = [1,1,1,2]
            channels = [96, 192, 384, 768]

        elif model_size == 'b':
            print("Using big model")
            blocks = [2,2,18,2]
            channels = [128, 256, 512, 1024]
        
        self.channels = channels
        norm = 'batch'
        bias = True
        self.n_blocks = sum(blocks)
        drop_path = 0.1
        stochastic = True
        epsilon = 0.2
        self.blocks = blocks

        self.num_patches = (128 // 4) * (1024// 4)

        dpr = [x.item() for x in torch.linspace(0,drop_path,self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k,k,self.n_blocks)]
        max_dilation = 128// max(num_knn)
        k = 9
        conv = 'mr'
        act = 'gelu'
        epsilon = 0.2 
        stochastic = True 
        bias = True
        freq_num = 128
        time_num = 1024
        
        self.stem = Stem_conv(1,self.channels[0],act=act)

        self.act = act
        reduce_ratios = [4,2,1,1]
        self.prediction = Seq(nn.Conv2d(channels[-1],1024,1,bias=True), nn.BatchNorm2d(1024), act_layer(act), nn.Dropout(0.1),nn.Conv2d(1024,200,1,bias=True))
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0],128 // 4,1024// 4))
        self.backbone = nn.ModuleList([])
        idx= 0
        self.HW =  self.num_patches

        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(DownSample(channels[i-1], channels[i]))
                self.HW = self.HW//4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=self.HW, drop_path=dpr[idx],
                                    relative_pos=False),
                          FFN(in_features=channels[i], hidden_features=channels[i] * 4, act=act, drop_path=dpr[idx])
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.model_init()

       
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.adj.size(1))
        self.adj.data.uniform_(-stdv, stdv)
        self.adj.data.fill_diagonal_(0)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    def forward(self,x):
        input_data = x
        
        input_data =input_data.unsqueeze(1)
        #device = input_data.get_device()

        B,C,H,W = input_data.shape
        input_data = input_data.transpose(2,3)
        x = self.stem(input_data) + + self.pos_embed

        for i in range(len(self.backbone)):
                x = self.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.prediction(x)
        x = x.squeeze(-1).squeeze(-1)
        preds = torch.sigmoid(x)
        
        print(preds[:1,:])
        preds = torch.rand(8,200).cuda()
        return preds






        




class AST(nn.Module):

    def __init__(self,k=9,act='gelu',norm='batch',bias=True,dropout=0.0,dilation=True,epsilon=0.2,drop_path=0.1,size ='m',
               num_class=200,emb_dims=1024,freq_num=128,time_num=1024):
        
        super().__init__()
        model_size = 'base384'
        pretrain_type = 'img'
        label_dim = 200
        fstride = 16
        tstride= 16
        input_fdim = 128
        input_tdim = 1024
        pretrain = False

        timm.models.vision_transformer.PatchEmbed = PatchEmbed_ast
        if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=True)
        elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=True)
        elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=True)
        elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)

        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        #f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
        if t_dim <= self.oringal_hw:

            new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.oringal_hw:

            new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
              
    
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)

        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        print(len(self.v.blocks))
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        
        x = self.mlp_head(x)
        
        x = torch.sigmoid(x)
        print(x[:1,:])
        return x
    
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim



        
       
    





        

    

        
        