from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import json
import os 
import pdb 
from torch.utils.data.distributed import DistributedSampler
from src.data.dataset import FSDDataset 



class AudioSetModule(LightningDataModule):

    def __init__(self,json_path:str,
                 data_dir:str,
                 meta_path:str,
                 label_csv_pth:str,
                 samplr_csv_pth:str,
                 balance_samplr:bool,
                 batch_size:int,
                num_workers:int,
                pin_memory:bool,
                persistent_workers:bool,
                sr:int,
                fmin:int,
                fmax:int,
                num_mels:int,
                window_type:str,
                target_len:int,
                freqm:int,
                timem:int, 
                mixup:float,
                norm_mean:float,
                norm_std:float,
                subset:str,
                num_devices: str,
                
                 )->None:
        
        super().__init__()
        self.batch_size = batch_size
        self.json_path = json_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sr = sr
        self.meta_path = meta_path
        self.fmax = fmax
        self.fmin = fmin
        self.num_mels = num_mels
        self.window_type = window_type
        self.target_len = target_len
        self.freqm = freqm
        self.timem = timem
        self.mixup = mixup
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.label_csv_pth = label_csv_pth
        self.sampler_csv_pth = samplr_csv_pth
        self.balance_samplr = balance_samplr
        self.data_dir = data_dir
        self.subset = subset
        if not os.path.exists(self.json_path):
            os.mkdir(self.json_path)
        self.num_devices = int(num_devices)
        if self.subset == 'bal':
            self.tr_json = self.json_path + 'audioset_bal_tr.json'
        else:
            self.tr_json = self.json_path + 'audioset_all_tr.json'
        self.eval_json = self.json_path + 'audioset_eval.json'
        
        self.audio_conf = {'sr':sr,'fmin':fmin,'fmax':fmax,'num_mels':num_mels,'window_type':window_type,'target_len':target_len,'freqm':freqm,'timem':timem,'norm_mean':norm_mean,'norm_std':norm_std,'mixup':mixup} 
        self.persistent_workers = persistent_workers
        
    
    
    def setup(self,stage: Optional[str] = None)-> None:
        
        self.tr_dataset = FSDDataset(self.tr_json,self.audio_conf,mode='train',label_csv=self.label_csv_pth)
        self.eval_dataset = FSDDataset(self.eval_json,self.audio_conf,mode='eval',label_csv=self.label_csv_pth)
    

    def train_dataloader(self)-> DataLoader[Any]:

        if self.balance_samplr == True:
            
            samples_weight = np.loadtxt(self.sampler_csv_pth,delimiter=',',dtype=np.float32)
            self.sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.train_dataset))
            #self.sampler = torch.utils.subset_random_sampler.SubsetRandomSampler(indices)
            shuffle_tr = False
        
        else:
            self.sampler = None
            shuffle_tr = True
        #self.sampler = DistributedSampler(self.train_dataset,shuffle=False) if self.num_devices > 1 else None
        #self.sampler = None
        return DataLoader(dataset=self.tr_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.sampler,
            persistent_workers=self.persistent_workers) 
    
    def val_dataloader(self)-> DataLoader[Any]:

        self.sampler = None

        return DataLoader(dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            sampler=self.sampler,
            persistent_workers=self.persistent_workers     
        )
    
   
    
    


        
        
    


            
        






            
        
        
        #print(cfg.path)
        
        #print(cfg.path)
    


        
        

            




         










    