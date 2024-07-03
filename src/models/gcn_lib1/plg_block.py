import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer,MLP
from .torch_edge import DenseDilatedKnnGraph,DenseDilatedKnnGraph_plg,DenseDilatedKnnGraph_new
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
from torch.nn import Sequential 
from models.model_utils import FFN 

class MRConv2d_plg(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d_plg, self).__init__()
        self.nn_plg = BasicConv([in_channels*2, in_channels], act, norm, bias)
        self.mlp = MLP([in_channels*2, in_channels], act, norm, bias)
    def forward(self, lab_x, edge_index, patch_x=None):
        
        
        
        x_i = batched_index_select(lab_x, edge_index[1])
        x_j = batched_index_select(patch_x, edge_index[0])
       
        
        #if y is not None:
         #   x_j = batched_index_select(y, edge_index[0])
        #else:
         #   x_j = batched_index_select(x, edge_index[0])
        x_i, _ = torch.max(x_i - x_j, -1, keepdim=True)
        b, c, n, _ = lab_x.shape
        lab_x = torch.cat([lab_x.unsqueeze(2), x_i.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        

        return  self.nn_plg(lab_x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphConv2d_plg(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d_plg, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv_plg = MRConv2d_plg(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv_plg(x, edge_index, y)


class DyGraphConv2d(GraphConv2d_plg):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph =  DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, lab_x,patch_x, relative_pos=None):
       # B, C, H, W = patch_x.shape
        B,C,H1,W1 = lab_x.shape
        #y = None
        
        #if self.r > 1:
         #   y = F.avg_pool2d(x, self.r, self.r)
          #  y = y.reshape(B, C, -1, 1).contiguous()            
        lab_x = lab_x.reshape(B, C, -1, 1).contiguous()
        
        patch_x = patch_x.reshape(B, C, -1, 1).contiguous()
        
        edge_index = self.dilated_knn_graph(lab_x, patch_x, relative_pos)
        
        x = super(DyGraphConv2d, self).forward(lab_x, edge_index, patch_x)
        
        return x.reshape(B, -1, H1, W1).contiguous()
def gen_adj_new(A):
    
    D = torch.pow(A.sum(1).float(), -0.5)
    
    D = torch.diag(D)
    adj = torch.matmul(D,torch.matmul(A, D))
    
    return adj
class NONLocal_lab(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NONLocal_lab, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):

        batch_size = x.size(0)
        theta_x = self.theta(x).squeeze(3).transpose(1,2)
        
        #theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        #print(theta_x.shape)
        #theta_x = theta_x.squeeze(2)
        phi_x = self.phi(x)
        phi_x = phi_x.squeeze(3)
        
        #phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        #print(phi_x.shape)
        #phi_x = phi_x.squeeze(2).transpose(0,1)
        #phi_x = phi_x.transpose(1,2)
        
        f = torch.matmul(theta_x,phi_x)
        
        
        N = f.size(-1)
        
        f_div_C = f / N

        return f_div_C

#class GCNResnet(nn.Module):
 #   def __init__(self, num_classes, in_channel=256):
  #      super(GCNResnet, self).__init__()
        
   #     self.num_classes = num_classes
     #   self.pooling = nn.MaxPool2d(14, 14)
    #    ffn_channel = 1024
      #  self.non_local_lab = NONLocal_lab(in_channel)
       # self.gc1 = GraphConvolution(in_channel, in_channel)
        #self.gc2 = GraphConvolution(2*in_channel,in_channel)
        #self.activate = nn.LeakyReLU(0.2)
        #self.adj = nn.Parameter(torch.Tensor(200, 200))
        
        #self.ffn = FFN(ffn_channel, ffn_channel * 4, act='gelu', drop_path=0.1)
        # image normalization
        #self.image_normalization_mean = [0.485, 0.456, 0.406]
        #self.image_normalization_std = [0.229, 0.224, 0.225]
    #def reset_parameters(self):
     #   stdv = 1. / math.sqrt(self.adj.size(1))
      #  self.adj.data.uniform_(-stdv, stdv)

    #def forward(self,inp):
        #feature = self.features(feature)
        #feature = self.pooling(feature)
        #feature = feature.view(feature.size(0), -1)
     #   adj = self.adj
      #  inp = inp.to('cuda')
       # Batch = inp.shape[0]
        
        #inp = inp[0]
        
        #adj = adj.unsqueeze(0).repeat(Batch,1,1)
        #inp1 = inp.unsqueeze(2)
        #inp1 = inp.transpose(1,2)
        #adj = self.non_local_lab(inp)
        
        #A = torch.eye(200, 200).float().cuda()
        #A1 = A.unsqueeze(0).repeat(Batch,1,1)
        #A1 = torch.autograd.Variable(A)
        #adj= adj+ A
        
        #adj = gen_adj_new(adj)
        #print(adj)
        #inp = inp.transpose(1,2).squeeze(3)
        #x = self.gc1(inp, adj)
        
        #x = self.activate(x)
        #x = self.gc2(x, adj)
        #x = torch.matmul(adj,inp)
        
        #out = x+inp
        #x = x.transpose(0, 1)
        #x = torch.matmul(feature, x)
        #A = A1.unsqueeze(0)
        #adj = adj.unsqueeze(0)
        #return x, adj, A
        #return out
def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class Grapher_plg(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher_plg, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        #self.gcn = GCNResnet(num_classes=200, in_channel=256)
        self.graph_conv = DyGraphConv2d(in_channels, in_channels*2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc4 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.emb = nn.Parameter(torch.rand(1, 200, 256))
        drop_path = 0.1
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        #self.adj = nn.Parameter(torch.rand(200, 200))
        self.ffn = FFN(in_features=in_channels, hidden_features=in_channels * 4, act=act, drop_path=0.1)
        self.adj = nn.Embedding(200,200)
        self.inp = torch.Tensor(np.arange(200)).view(1,-1).long()
        #self.adj.weight.data = 1.0 - torch.eye(self.adj.weight.data.size(0))
        
        #self.adj_new.fill_diagonal_(0.0,wrap=True)
        #self.channels  = [[80, 160, 400, 640]]
        #self.conv_patch = nn.ModuleList([])
        #for i in range(len(self.channels)):
        #    self.conv_patch+= [Sequential(nn.Conv2d(self.channels[i],200,1,bias=True))]
        #self.conv_patch = Sequential (*self.conv_patch)
        if relative_pos:
            
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            #print(relative_pos_tensor.shape)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)
        #self.reset_parameters()
        self.adj.apply(weights_init)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.adj.size(1))
        self.adj.data.uniform_(-stdv, stdv)
        self.adj.data.fill_diagonal_(0)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self,x):
        #_tmp = x
        #x = inp_dict[0]
        #adj = inp_dict[1]
        #adj = adj.unsqueeze(3)
        
        #index = self.channels.index(C)
        #x_copy = x
        #x_copy = self.conv_patch[index](x_copy)
        #adj = self.adj_new.cuda()
        #emb_plg = self.emb.unsqueeze(3).repeat(B,1,1,1).transpose(1,2).cuda()
        #emb_plg = torch.cat((emb_plg,x_copy),dim=2)
        #adj_bl = self.adj 
        #A = torch.eye(200, 200).float().cuda()
        #A1 = A.unsqueeze(0).repeat(Batch,1,1)
        #A1 = torch.autograd.Variable(A)
        #adj= adj+ A
        #adj = gen_adj_new(adj)
        #lab_x = inp_dict[0]
        #patch_x = inp_dict[1]
        #B,C,_,_ = x.shape
        #x = self.fc3(x)
       
        #x = self.ffn(x)
        
        #const_label_input = self.inp.repeat(x.size(0),1).cuda()
        #adj = self.adj(const_label_input)
        
        #adjj = self.adj(self.inp)
        #adj = adj.unsqueeze(0).repeat(B,1,1)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        patch_x = x[:,:,200:,:]
        
        lab_x = x[:,:,:200,:]
        _tmp = lab_x
        
        #adj = self.adj.unsqueeze(0).repeat(B,1,1)
        
        lab_x = self.graph_conv(lab_x,patch_x,relative_pos=None)
        #lab_x = self.fc4(lab_x)
        lab_x = self.drop_path(lab_x) + _tmp
        
        #lab_x = lab_x.transpose(1,2).squeeze(-1)
        
        #out1 = torch.matmul(adj,lab_x)
        #out1 = out1.transpose(1,2).unsqueeze(-1)
        #lab_x = lab_x.transpose(1,2).unsqueeze(-1)
        #lab_x = lab_x +out1
        
        lab_x = self.ffn(lab_x)
        
        return lab_x
        #
       # _tmp = emb_plg[:,:,:200,:]
        #emb_plg = self.fc3(emb_plg)
        
        
        
        
       
        
        
        
        
        
        
        #_tmp1 = lab_x
        
        #lab_x = torch.matmul(self.adj,lab_x)
       
        #np.savetxt('/homes/ss380/deeplearn/graph_exp/backup_graphexp/Spec_GNN/AST_model/psla-main/src/models/my_file.txt', self.adj.cpu().detach().numpy())
        #lab_x = lab_x.transpose(1,2).unsqueeze(3)
        #lab_x = lab_x + _tmp1
        #lab_x = self.ffn(lab_x)
        #lab_x_corr = torch.matmul(self.adj,lab_x)  +lab_x
               
        #lab_plg = x[:,:,:200,:]
        
        #lab_x_corr = self.gcn(lab_x)
        
        #lab_x_corr = lab_x_corr.transpose(1,2).unsqueeze(3)
       
        #lab_plg = lab_plg +x_lab
        #x[:,:,:200,:] = lab_x
        
        