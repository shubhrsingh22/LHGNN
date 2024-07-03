import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn import Sequential as Seq
from models.components.hgcn_lib.torch_nn import BasicConv,batched_index_select


def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class HyperedgeConstruction(nn.Module):

    def __init__(self,in_channels,cluster_type):

        super(HyperedgeConstruction,self).__init__()
        self.in_channels = in_channels
        self.cluster_type = cluster_type
        
        if self.cluster_type == 'cosine':
            self.sim_alpha = nn.Parameter(torch.ones(1))
            self.sim_beta = nn.Parameter(torch.zeros(1))
        
        elif self.cluster_type == 'soft-kmeans':
            self.num_iter = 1
        
        
    
    def forward(self,x,centroids):
        """
        Args:   
        Inputs:
            x: (B,C,H,W) Input feature map
            centroids: (B,C,num_centroids) Pooled centroids
        Outputs:
            hyperedges: (B,C,num_centroids) Hyperedge features
            similarity: (B,H*W,1,num_centroids) Similarity matrix
            calculated using cosine similarity/soft-kmeans/attention    
        """

        b,c,h,w = x.shape
        
        
        if self.cluster_type == 'cosine':
            x_r = x.reshape(b,c,-1)
            similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(centroids.reshape(b,c,-1).permute(0,2,1), x_r.permute(0,2,1)))
            # similarity matrix shape (B,num_centroids,H*W)
            _, max_idx = similarity.max(dim=1, keepdim=True)
            mask = torch.zeros_like(similarity)
            mask.scatter_(1, max_idx, 1.)
            
            
            similarity= similarity*mask
            hyperedge_agg= ( x_r.permute(0,2,1).unsqueeze(dim=1)*similarity.unsqueeze(dim=-1) ).sum(dim=2)
            
            hyperedges = ( hyperedge_agg + centroids.permute(0,2,1))/ (mask.sum(dim=-1,keepdim=True)+ 1.0)
            return hyperedges,similarity
            
        elif self.cluster_type == 'soft-kmeans':
            
            x = x.reshape(b,h*w,c)
            m =2 
            b,c,num_centroids = centroids.shape
            with torch.no_grad():
                centroids = centroids.detach()
                #centroids = torch.randn((b, c, num_centroids), device=x.device, dtype=x.dtype)
                
                for i in range(self.num_iter):
                    dist_to_centers = torch.cdist(x, centroids.transpose(1, 2))
                    inv_dist = 1.0 / (dist_to_centers + 1e-10)
                    power = 2 / (m - 1)
                    membership = (inv_dist / inv_dist.sum(dim=-1, keepdim=True)).pow(power)
                    weights = membership.pow(m).unsqueeze(2)
                    centroids = torch.sum(weights * x.unsqueeze(-1), dim=1) / weights.sum(dim=1)
                    hyperedges = centroids.clone()
                return hyperedges,weights


                
                
    

class RefineAttention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        B, N, C = x.shape
        #N = H * W
                
        q, k, v = self.qkv(x.permute(0,2,1).unsqueeze(-1)).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn)
        x = x.reshape(B,C,N,-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MaxGraphConv(nn.Module):


    def __init__(self,in_channels,out_channels,k_g):

         super(MaxGraphConv,self).__init__()
         self.in_channels = in_channels
         self.out_channels = out_channels
         self.k_g = k_g
         self.nn = BasicConv([in_channels*2, out_channels], act='gelu', norm='batch', bias=True)

    def forward(self,x):
        """
        Args:
        Inputs:
            x: (B,C,num_centroids)
        Outputs:
            x: (B,C*2,num_centroids)
        """
        
        b,c,n_points = x.shape

        with torch.no_grad():
            x = x.permute(0,2,1) # (B,num_centroids,C)
            x = F.normalize(x,dim=-1)
            dist = torch.cdist(x,x) # (B,num_centroids,num_centroids)
            eye = torch.eye(dist.size(1), device=dist.device).unsqueeze(0) * 1e6
            dist = dist + eye
            
            _,nn_idx = torch.topk(-dist,k=self.k_g,dim=-1) #
            # nn_index shape: (B,num_centroids,k_g)
            center_idx = torch.arange(0, n_points, device=x.device).repeat(b, self.k_g, 1).transpose(2, 1)
            # center idx shape: (B,num_centroids,k_g)
            #edge_index = torch.stack((nn_idx, center_idx), dim=0)

        
            #batch,channel,num_centroids,_ = x.permute(0,2,1).unsqueeze(-1)

            batch_indices = torch.arange(b).unsqueeze(1).unsqueeze(2).expand(-1, n_points, self.k_g)
            #batch indices shape: (B,num_centroids,k_g)
            nearest_features = x[batch_indices, nn_idx, :]
            # nearest_features shape: (B,num_centroids,k_g,C)
            
            diff = torch.abs(x.unsqueeze(2) - nearest_features)
            max_diff,_ = torch.max(diff,dim=2,keepdim=True)
            #  max_diff shape: (B,num_centroids,1,C)
            max_diff = max_diff.permute(0,3,1,2)
            # max_diff shape: (B,C,num_centroids,1)

            x = x.permute(0,2,1).unsqueeze(-1)
           # x shape : B,C,num_centroids,1
            batch,channel,num_centroids,_ = x.shape
            x = torch.cat([x.unsqueeze(2),max_diff.unsqueeze(2)],dim=2).reshape(batch,2*channel,num_centroids,-1)

            # x shape: (B,C*2,num_centroids,1)
            return self.nn(x)
           


    #    with torch.no_grad():
     #       x = x.permute(0,2,1)
      #      x = F.normalize(x,dim=-1)
       #     dist = torch.cdist(x,x)
        #    _,nn_index = torch.topk(-dist,k=self.k_g,dim=-1,largest=False)
         #   # nn_index shape: (B,num_centroids,k_g)
          #  center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
           # index = index.unsqueeze(dim=1)
            #x = x.unsqueeze(dim=-1)
            #x = x.gather(-2,index).squeeze(dim=1)
            #x = x.permute(0,2,1)
            #x = x.reshape(x.shape[0],-1,x.shape[2])
        

    
    
