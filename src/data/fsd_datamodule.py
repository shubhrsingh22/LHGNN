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



class FSDDataModule(LightningDataModule):

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
        if not os.path.exists(self.json_path):
            os.mkdir(self.json_path)
        self.num_devices = int(num_devices)
        self.train_json = self.json_path + 'fsd_tr_full.json'
        self.val_json = self.json_path + 'fsd_val_full.json'
        self.eval_json = self.json_path + 'fsd_eval_full.json'
        self.audio_conf = {'sr':sr,'fmin':fmin,'fmax':fmax,'num_mels':num_mels,'window_type':window_type,'target_len':target_len,'freqm':freqm,'timem':timem,'norm_mean':norm_mean,'norm_std':norm_std,'mixup':mixup} 
        self.persistent_workers = persistent_workers
        
    def process_metafiles(self,meta_csv,set_name):


        if set_name == 'train':

            tr_files = []
            va_files = []
            tr_cnt, va_cnt = 0, 0


            for i in range(len(meta_csv)):
                
                try:
                    fileid = meta_csv[i].split(',"')[0]
                    labels = meta_csv[i].split(',"')[2][0:-1]
                    set_info = labels.split('",')[1]
                
                except:
                    fileid = meta_csv[i].split(',')[0]
                    labels = meta_csv[i].split(',')[2]
                    set_info = meta_csv[i].split(',')[3][0:-1]
                
                labels = labels.split('",')[0]
                label_list = labels.split(',')
                new_label_list = []
                for label in label_list:
                    new_label_list.append(label.strip('"'))
                new_label_list = ','.join(new_label_list)

                wav_file_path = os.path.join(self.data_dir,'dev_audio/',str(fileid)+'.wav')
                cur_dict = {"wav":wav_file_path,"labels":new_label_list}
                if set_info == 'trai':
                    tr_files.append(cur_dict)
                    tr_cnt += 1
                elif set_info == 'va':
                    va_files.append(cur_dict)
                    va_cnt += 1
                
                else:
                    raise ValueError("Invalid set info")
            print("Total number of training files: ",tr_cnt)
            print("Total number of validation files: ",va_cnt)
            return tr_files,va_files,tr_cnt,va_cnt
        
        else: 

            eval_files = []
            eval_cnt = 0

            for i in range(len(meta_csv)):

                try:
                    fileid = meta_csv[i].split(',"')[0]
                    labels = meta_csv[i].split(',"')[2][0:-1]

                except:
                    fileid = meta_csv[i].split(',')[0]
                    labels = meta_csv[i].split(',')[2]
                
                label_list = labels.split(',')
                new_label_list = []

                for label in label_list:
                    new_label_list.append(label)
                
                if len(new_label_list) != 0:
                    new_label_list = ','.join(new_label_list)
                    cur_dict = {"wav":os.path.join(self.data_dir,'eval_audio/',str(fileid)+'.wav'),"labels":new_label_list}
                    eval_files.append(cur_dict)
                    eval_cnt += 1
            print("Total number of evaluation files: ",eval_cnt)
            return eval_files,eval_cnt
        
                
                
    # def prepare_data(self)-> None:
        
        

    #     if not os.path.exists(self.train_json) :

    #         print("Preparing JSON files for FSD")
            
    #         dev_csv = np.loadtxt(os.path.join(self.meta_path,'dev.csv'),skiprows=1,dtype=str)
    #         eval_csv = np.loadtxt(os.path.join(self.meta_path,'eval.csv'),skiprows=1,dtype=str)
    #         tr_files,va_files,tr_cnt,va_cnt = self.process_metafiles(dev_csv,'train')
    #         eval_files,eval_cnt = self.process_metafiles(eval_csv,'eval')
    #         with open (self.train_json,'w') as f:
    #             json.dump({'data':tr_files},f,indent=1)
    #         with open (self.val_json,'w') as f:
    #             json.dump({'data':va_files},f,indent=1)
    #         with open (self.eval_json,'w') as f:
    #             json.dump({'data':eval_files},f,indent=1)
    #         import pdb; pdb.set_trace()
    #         print("JSON files saved")
    
    def setup(self,stage: Optional[str] = None)-> None:
        
        self.train_dataset = FSDDataset(self.train_json,self.audio_conf,mode='train',label_csv=self.label_csv_pth)
        
        
        self.val_dataset = FSDDataset(self.val_json,self.audio_conf,mode='val',label_csv=self.label_csv_pth)
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
        return DataLoader(dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,drop_last=True) 
    
    def val_dataloader(self)-> DataLoader[Any]:

        self.sampler = DistributedSampler(self.val_dataset,shuffle=False) if self.num_devices > 1 else None

        return DataLoader(dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,drop_last=True     
        )
    
    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader. """
        #self.sampler = DistributedSampler(self.eval_dataset_dataset,shuffle=False) if self.num_devices > 1 else None
        
        return DataLoader( dataset=self.eval_dataset,  batch_size=self.batch_size,  num_workers=self.num_workers,  pin_memory=self.pin_memory,  shuffle=False,persistent_workers=self.persistent_workers)
    
    


        
        
    


            
        






            
        
        
        #print(cfg.path)
        
        #print(cfg.path)
    


        
        

            




         










    