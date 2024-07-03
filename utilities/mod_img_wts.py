import torch 
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/homes/ss380/deeplearn/graphtrain/hypergraph/hypergraph_exps/HGANN/')
from src.models.components.Hypergraph import HGCN




freq_num = 128
time_num = 1024
last_model_pth = '/import/research_c4dm/ss380/hypergraph_exps/audio_tagging/pretrain_ckpt/last.pth.tar'
best_model_pth = '/import/research_c4dm/ss380/hypergraph_exps/audio_tagging/pretrain_ckpt/model_best.pth.tar'



def modify_img_wts_audioset():
    model = HGCN(num_class=200)
    state_dict = torch.load(best_model_pth)
    state_dict = state_dict['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():


        if name == 'stem.convs.0.weight':
            
            param = torch.mean(param,dim=1,keepdim=True)
            

        if 'pos_embed' in name:

            param = F.interpolate(param, size=(freq_num//4,time_num//4), mode='bicubic', align_corners=False)
                        
        elif 'relative_pos' in name:
            
            target_shape = own_state[name].shape[-2:]
            
            h = own_state[name].shape[1]
            w = own_state[name].shape[2]
            

            param = F.interpolate(param.unsqueeze(1), size=(h,w), mode='bicubic', align_corners=False).squeeze(1)
        
        

            
            #h = own_state[name].shape[1]
            #w = own_state[name].shape[2]
            #param = F.interpolate(param.unsqueeze(1), size=(h,w), mode='bicubic', align_corners=False).squeeze(1)


            
        if name in own_state.keys():
            own_state[name].copy_(param)
            print(f"loaded weights for:{name} ")
        
        
    
    torch.save(own_state,'/import/research_c4dm/ss380/hypergraph_exps/audio_tagging/pretrain_ckpt/img2fsd.pth.tar')
        #model.load_state_dict(own_state)
    
    
    
    
        

        

        



        
        























if __name__ == "__main__":
    modify_img_wts_audioset()
    #model = HGCN(num_class=200)
    #path = '/import/research_c4dm/ss380/hypergraph_exps/audio_tagging/pretrain_ckpt/img2fsd.pth.tar'
    #state_dict = torch.load(path)
    
   # modify_img_wts_fsd()