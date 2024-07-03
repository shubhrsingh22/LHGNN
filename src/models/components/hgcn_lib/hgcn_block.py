import numpy as np
import torch
from torch import nn
#from torch_nn import BasicConv, batched_index_select, act_layer,batched_index_select_tmp
from models.components.hgcn_lib.pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn import Sequential as Seq
from models.components.hgcn_lib.torch_nn import DpcKnn,FFN,BasicConv,act_layer,MLP,batched_index_select
from models.components.hgcn_lib.torch_vertex import HyperedgeConstruction,RefineAttention,MaxGraphConv

import time

class HyperGraphConv(nn.Module):

    def __init__(self,in_channels, out_channels, num_centroids, dilation, conv,k_dpc,
                                         cluster_type,refine_hyperedge,act,norm,bias,stochastic,epsilon,pool_win,centroid_ratio=0.25):
        super(HyperGraphConv,self).__init__()
        self.get_centroids = DpcKnn(num_centroids,k_dpc,in_channels=in_channels)
        self.refine_hyperedge = refine_hyperedge
        num_centroids = 25
        if refine_hyperedge == 'attn':
            self.refine_cluster = RefineAttention(in_channels,num_heads=1)
        if refine_hyperedge == 'graph':
            self.refine_cluster = MaxGraphConv(in_channels=in_channels,out_channels=in_channels*2,k_g =num_centroids-1)
        self.nn_hyperedge = BasicConv([in_channels, in_channels], act, norm, bias)
        self.get_hyperedges = HyperedgeConstruction(in_channels,cluster_type)
        self.center_ffn = FFN(in_channels,in_channels*4,act=act)
        self.pool_win = pool_win
        self.centers_proposal = nn.AdaptiveAvgPool2d((10,10))
        self.centroid_ratio = centroid_ratio
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)
    def forward(self,x,relative_pos,y=None):
        b,c,h,w = x.shape
        x_flat = x.reshape(b,c,-1)
        n_points = x_flat.shape[2]
        num_points = y.shape[2]
        num_centroids = int(self.centroid_ratio * num_points)
        #c_h = c_w = self.pool_win
        #pad_l = pad_t = 0
        #pad_r = (c_w - w % c_w) % c_w
        #pad_b = (c_h - h % c_h) % c_h
        #if pad_r > 0 or pad_b > 0:
        #    x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        #else:
        #    x_pad = x
        
        #hh = h//c_h
        #ww = w//c_w
        
        #pooled_centroids = F.adaptive_avg_pool2d(x, (hh, ww))
        #start_time = time.time()
        centroids = self.get_centroids(y,relative_pos,num_centroids)
        
        #centroids shape: Batch, channel, num_centroids
        
        
        agg_hyperedges,assign_matrix = self.get_hyperedges(x,centroids)
        
        # agg_hyperedges shape: (B,C,num_centroids)
        # assign_matrix shape: (B,H*W,1,num_centroids)
        

        assign_matrix = assign_matrix.squeeze(-2) 
        
        # assign_matrix shape: Batch, H*W, num_centroids
        #batch_indices = torch.arange(b).view(b, 1, 1).expand(-1, h*w, 9)
        _, nn_idx = torch.topk(assign_matrix, k=5, largest=True, dim=-1)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(b, 5, 1).transpose(2, 1)
        edge_idx = torch.stack((nn_idx, center_idx), dim=0)

        # center_idx shape: Batch, H*W, 9
        # nearest_indices shape: Batch, H*W, 9
        #nearest_indices_expanded = nearest_indices.unsqueeze(1).expand(-1, c, -1, -1)
        # nearest_indices shape: Batch, H*W, 9, num_centroids
        #print(f'nearest indices shape:{nearest_indices.shape}')

        #agg_transpose = agg_hyperedges.transpose(1,2)
        #gathered_agg_features = agg_transpose[batch_indices, nearest_indices, :].transpose(2, 3)
        #print(f'gathered_agg_features shape:{gathered_agg_features.shape}')
        # gathered_agg_features shape: Batch, H*W, C, 9


       # nearest_hyperedge_features = agg_hyperedges.transpose(1,2)[nearest_indices]
        
        # nearest_hyperedge_features shape: Batch, H*W, 9, num_centroids
        
        #nearest_hyperedges = torch.gather(agg_hyperedges.unsqueeze(-2).expand(-1, -1, -1, 9, -1), -2, nearest_indices)
        # nearest_hyperedges shape: Batch, H*W, 9, num_centroids, C

        #_,max_idx = assign_matrix.max(dim=-1,keepdim=True)
        # max_idx shape: Batch, H*W, 1
        #mask = torch.zeros_like(assign_matrix)
        # mask shape: Batch, H*W, num_centroids
        #mask.scatter_(-1,max_idx,1)
        #assign_matrix = assign_matrix * mask
        #assign_matrix = mask
        #cluster_hyperedges = torch.bmm(x_flat,assign_matrix)
        #agg_hyperedges = centroids + agg_hyperedges
        agg_hyperedges = agg_hyperedges+ self.center_ffn(agg_hyperedges.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        # cluster_hyperedges shape: Batch, channels, num_centroids
        #agg_hyperedges = torch.cat([cluster_hyperedges,cluster_hyperedges-agg_hyperedges],dim=1)
        #print(f'hyperedge shape:{agg_hyperedges.shape}')
        # agg_hyperedges shape: Batch, channels, num_centroids
        
        #agg_hyperedges = 

        if self.refine_hyperedge == 'graph':
            
            refined_hyperedges = self.refine_cluster(agg_hyperedges)
            # refined hyperedges shape: Batch, channels *2, num_centroids,1
            refined_hyperedges = refined_hyperedges.permute(0,2,1,3).squeeze(-1)
            # refined hyperedges shape: Batch, num_centroids, channels *2
            
            x_projected = torch.bmm(assign_matrix,refined_hyperedges)
            # x_projected shape : Batch, H*W, channels *2

            x_projected = x_projected.permute(0,2,1).unsqueeze(dim=-1)
        
            return x_projected,refined_hyperedges
        else: 
            refined_hyperedges = agg_hyperedges
            #refined_hyperedges = self.nn_hyperedge(refined_hyperedges.unsqueeze(dim=-1)).squeeze(dim=-1)
            refined_hyperedges = refined_hyperedges.unsqueeze(dim=-1)
            x_flat = x_flat.unsqueeze(-1)
            x_j =  batched_index_select(refined_hyperedges, edge_idx[0])
            x_i = batched_index_select(x_flat, edge_idx[1])
            x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
            
            b, c, n, _ = x_flat.shape
            x = torch.cat([x_flat.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
            
            x = self.nn(x)
            return x,refined_hyperedges

            #refined_hyperedges = refined_hyperedges.permute(0,2,1)

            
            # refined hyperedges shape: Batch, num_centroids, channels
            #nearest_indices = nearest_indices.unsqueeze(1).expand(-1, c, -1, -1)
            # nearest_indices shape: Batch, H*W, 9, num_centroids

            #batch_indices = torch.arange(b).view(-1, 1, 1, 1).expand(-1, c, h*w, 9)
            # batch_indices shape: Batch, channels, H*W, 9

            #gathered_centroid_features = refined_hyperedges[batch_indices, nearest_indices, torch.arange(c).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(b, -1, h*w, 9)]
            
            # gathered_centroid_features shape: Batch, channels, h*w, 9
            
            #gathered_centroid_features = gathered_centroid_features.permute(0,2,3,1)
            # gathered_centroid_features shape: Batch, H*W, 9, channels
            #gathered_centroid_features = torch.randn(b,h*w,9,c).cuda()
            #x_i = x.reshape(b,h*w,c).unsqueeze(2)
            # x_i shape: Batch, H*W, 1, channels
            
            
            #features_difference = x_i - gathered_centroid_features

            

            # features_difference shape: Batch, H*W, 9, channels
            
            # max_features_difference shape: Batch, H*W, 1, channels
            #max_features_difference,_ = torch.max(features_difference, dim=-2, keepdim=True)

            
            #x_i = torch.cat([x_i,max_features_difference],dim=2)
            #b,n,_,c = x_i.shape
         
            #x_i = x_i.reshape(b,2*c,n,1)
            
            #x_projected = self.nn(x_i)
            
            # x_projected shape: Batch, channels * 2, H*W, 1

            #x_projected = torch.bmm(assign_matrix,refined_hyperedges)
            # x_projected shape : Batch, H*W, channels
            #x_projected = x_projected.permute(0,2,1).unsqueeze(dim=-1)
            # x_projected shape : Batch, channels, H*W, 1
            #x_flat = x_flat.unsqueeze(-1)
            #x_projected = self.nn(torch.cat([x_flat,x_projected],dim=1))
            #return x_projected,refined_hyperedges
        
            #refined_hyperedges = self.nn_hyperedge(agg_hyperedges.unsqueeze(dim=-1)).squeeze(dim=-1)
            # assign_matrix shape: Batch, H*W, num_centroids
        
        
        
        
        

        #if self.refine_cluster:
        #    refined_hyperedges = self.refine_cluster(agg_hyperedges)
        #else:
        #    refined_hyperedges = agg_hyperedges.permute(0,2,1).unsqueeze(dim=-1)
        #refined_hyperedges = self.nn_hyperedge(refined_hyperedges).permute(0,2,1,3).squeeze(-1)
        #end_time = time.time()
        #print(f'refine_cluster time: {end_time-start_time}')
        #refined_hyperedges = refined_hyperedges.permute(0,2,1,3).squeeze(-1)
        #x_projected = (torch.mul(refined_hyperedges.unsqueeze(dim=2),assign_matrix.unsqueeze(dim=-1))).sum(dim=1)
        
        #x_projected = self.nn_hyperedge(x_projected.reshape(b,c,h*w).unsqueeze(dim=-1))
        

        #x_projected = x_projected.reshape(b,c*2,h,w)
        
        

class HG_block(nn.Module):

    def __init__(self,in_channels,centroid_ratio,pool_win,dilation,conv, act, 
                norm,k_dpc,cluster_type,refine_cluster,bias,stochastic,epsilon,n,drop_path,
                relative_pos,reduce_ratio):
        super(HG_block,self).__init__()
        self.in_channels = in_channels
        self.centroid_ratio = centroid_ratio
        self.dilation = dilation
        self.conv = conv
        self.act = act
        self.norm = norm
        self.k_dpc = k_dpc
        self.cluster_type = cluster_type
        self.bias = bias
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.n = n
        self.num_centroids = int(self.centroid_ratio * self.n)
        self.pool_win = pool_win
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        r = 1
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)
        self.graph_conv = HyperGraphConv(in_channels, in_channels * 2, self.num_centroids, dilation, conv,k_dpc,
                                         cluster_type,refine_cluster,act,norm,bias,stochastic,epsilon,pool_win,self.centroid_ratio)
        self.r = reduce_ratio                          
    
    def forward(self,x):
        _tmp = x
        x = self.fc1(x)
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
        else:
            y = x
        B, C, H, W = x.shape
        y = y.reshape(B, C, -1, 1).contiguous() 
        
        relative_pos = self.relative_pos
        x,hyperedges = self.graph_conv(x,relative_pos,y=y)
        #x_new = torch.randn(4*1024, 4*1024, device="cuda")
        #x = torch.cat([x,x],dim=1)
        
        x = self.fc2(x)
        x = x.reshape(B,C,H,W)
        x = self.drop_path(x) + _tmp
        
        return x
        



                                    