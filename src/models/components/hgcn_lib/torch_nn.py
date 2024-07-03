# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d
from timm.models.layers import DropPath

##############################
#    Basic layers
##############################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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


class DpcKnn(nn.Module):

    def __init__(self, num_centroids,k_dpc,in_channels=None):
        super(DpcKnn, self).__init__()
        self.num_centroids = num_centroids  
        self.k_dpc = k_dpc
        self.num_centroids = 50
        #self.ffn = FFN(in_channels,hidden_features=in_channels*4,out_features=in_channels,act='gelu',drop_path=0.0)

    def forward(self,x,relative_pos,num_centroids):

        with torch.no_grad():
            
            B,C,H,W = x.shape 
        
            x = x.view(B, H*W, C)
          # use view instead of reshape
            batch_size, num_points, num_dims = x.shape
            
            dist_matrix = torch.cdist(x, x)/ (C ** 0.5)
            #if relative_pos is not None:
            #   dist_matrix += relative_pos
            
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=self.k_dpc, dim=-1, largest=False)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            density += torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
            mask = density[:, None, :] > density[:, :, None]
           # dist_max = dist_matrix.view(batch_size, -1).max(dim=-1)[0][:, None, None]
           # mask_bool = mask.bool()

# Use torch.where to select elements from dist_matrix or dist_max based on mask
           # dist_masked = torch.where(mask_bool, dist_matrix, dist_max)

# Now find the minimum
           # dist, index_parent = dist_masked.min(dim=-1)
            #mask = mask.type(x.dtype)
            #dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            with torch.cuda.amp.autocast():
                score = dist * density
                _, index_down = torch.topk(score, k=num_centroids, dim=-1)
                centers = x[torch.arange(x.size(0)).unsqueeze(1), index_down]
                centers = centers.transpose(1,2)
                #centers = self.ffn(centers.unsqueeze(-1)).squeeze(-1)
            
            
            #dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            #score = dist * density
            #_, index_down = torch.topk(score, k=self.num_centroids, dim=-1)
            
            
        return centers



def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature
        
    
    
        
def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    # Replace -1s in idx with the index pointing to the dummy node/hyperedge
    idx[idx == -1] = x.size(2) - 1  # Make sure this points to the last node/hyperedge which is the dummy one
    
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature       
        
        
        

    
    


    
def dpc_knn(x,k_dpc,relative_pos):

    B,C,H,W = x.shape 
        
    x = x.view(B, H*W, C)
          # use view instead of reshape
    batch_size, num_points, num_dims = x.shape
    dist_matrix = torch.cdist(x, x)
    if relative_pos is not None:
        dist_matrix += relative_pos
    
    dist_nearest, index_nearest = torch.topk(dist_matrix, k=k_dpc, dim=-1, largest=False)
    
