import numpy as np 
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader    
import random 
import torchaudio
import unittest
import json 
import csv
import h5py 

def label_to_index(label_csv):
    index_dict = {}

    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index_dict[row['mid']]  = int(row['index'])
        return index_dict


class FSDDataset(Dataset):

    def __init__(self,data_json,conf,mode=None,label_csv=None):
        super().__init__()
        with open(data_json, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.conf = conf
        self.mode = mode
        self.mixup = conf['mixup']
        self.index_dict = label_to_index(label_csv)
        self.num_labels = len(self.index_dict)
        
        self.num_mels = conf['num_mels']
        self.fmin = conf['fmin']
        self.fmax = conf['fmax']
        self.sr = conf['sr']
        self.window_type = conf['window_type']
        self.target_len = conf['target_len']
        self.freqm = conf['freqm']
        self.timem = conf['timem']
        self.norm_mean = conf['norm_mean']
        self.norm_std = conf['norm_std']
    
    def __len__(self):
        return len(self.data)
    
    def wav_2_fbank(self,filename,filename2=None):
        sr = 16000
        if filename2 is  None:
            
            #waveform,sr = torchaudio.load(filename)
            waveform = filename
            if waveform.dim() == 1:
             # If the waveform is 1D, add an extra dimension to make it 2D
                waveform = waveform.unsqueeze(0)
            #transform = torchaudio.transforms.Resample(sr,self.sr)
            #waveform = transform(waveform)
            waveform = waveform - waveform.mean()
            
        else:
            
            #waveform1,sr = torchaudio.load(filename)
            waveform1 = filename
            if waveform1.dim() == 1:
             # If the waveform is 1D, add an extra dimension to make it 2D
                waveform1 = waveform1.unsqueeze(0)
            #waveform2,sr = torchaudio.load(filename2)
            waveform2 = filename2
            if waveform2.dim() == 1:
             # If the waveform is 1D, add an extra dimension to make it 2D
                waveform2 = waveform2.unsqueeze(0)
            #transform = torchaudio.transforms.Resample(sr,self.sr)
            #waveform1 = transform(waveform1)
            #waveform2 = transform(waveform2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()
            
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1,waveform1.shape[1])
                    temp_wav[0,0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0,0:waveform1.shape[1]]
            
            # sample lambda from uniform distribution
            mix_lambda = np.random.beta(10,10)
            mix_waveform = mix_lambda * waveform1 + (1-mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform,htk_compat=True,sample_frequency=self.sr,use_energy=False,window_type='hanning',num_mel_bins=self.num_mels,dither=0.0,frame_shift=10)

        target_length = self.target_len
        n_frames = fbank.shape[0]
        if n_frames < target_length:
            p = target_length - n_frames
            m = torch.nn.ZeroPad2d((0,0,0,p))
            fbank = m(fbank)
        elif n_frames > target_length:
            fbank = fbank[0:target_length,:]
        
        if filename2 == None:
            return fbank,0
        else:
            return fbank,mix_lambda

    
    def __getitem__(self,idx):
        
        if self.mode == 'train':
            if random.random() < self.mixup:
                # do mixup
                
                mixup_idx = random.randint(0,len(self.data)-1)
                with h5py.File(self.data[idx]['wav'], 'r') as f1, h5py.File(self.data[mixup_idx]['wav'], 'r') as f2:
                    wav_file1 = torch.from_numpy(f1['audio'][:])
                    wav_file2 = torch.from_numpy(f2['audio'][:])
                     
                #wav_file1 = self.data[idx]['wav']
                #wav_file2 = self.data[mixup_idx]['wav']
                fbank,mix_lambda = self.wav_2_fbank(wav_file1,wav_file2)
                label = self.data[idx]['labels']
                label2 = self.data[mixup_idx]['labels']
                label_list = np.zeros(self.num_labels)
                for label_str in label.split(','):
                    label_list[self.index_dict[label_str]] += mix_lambda
                for label_str in label2.split(','):
                    label_list[self.index_dict[label_str]] += (1-mix_lambda)
                label_list = torch.FloatTensor(label_list)
            
            else:
                # no mixup
                with h5py.File(self.data[idx]['wav'], 'r') as f1:
                    wav_file = torch.from_numpy(f1['audio'][:])
                #wav_file = self.data[idx]['wav']
                
                #wav_file = self.data[idx]['wav']
                fbank,_ = self.wav_2_fbank(wav_file)
                label = self.data[idx]['labels']
                label_list = np.zeros(self.num_labels)
                for label_str in label.split(','):
                    label_list[self.index_dict[label_str]] += 1
                label_list = torch.FloatTensor(label_list)
            
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank,0,1)
            fbank = fbank.unsqueeze(0)
            fbank = freqm(fbank)
            fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank,0,1)
            fbank = (fbank - self.norm_mean)/self.norm_std
            return fbank,label_list
        else:
            with h5py.File(self.data[idx]['wav'], 'r') as f1:
                    wav_file = torch.from_numpy(f1['audio'][:])
            #wav_file = self.data[idx]['wav']
            fbank,_ = self.wav_2_fbank(wav_file)
            label = self.data[idx]['labels']
            label_list = np.zeros(self.num_labels)
            for label_str in label.split(','):
                label_list[self.index_dict[label_str]] += 1
            label_list = torch.FloatTensor(label_list)
            fbank = (fbank - self.norm_mean)/self.norm_std
            return fbank,label_list
                
                





#if __name__ == '__main__':
    
 #   audio_conf = {'sr':16000,'fmin':20,'fmax':8000,'num_mels':128,'window_type':'hanning','target_len':1024,'freqm':10,'timem':10,'norm_mean':-4.5,'norm_std':4.5,'mixup':0.5} 

  #  train_json = json.load(open('/homes/ss380/deeplearn/hypergraph/hypergraph_exps/hyper_exps/prep_data/datafiles/train_files.json','r'))
   # val_json = json.load(open('/homes/ss380/deeplearn/hypergraph/hypergraph_exps/hyper_exps/prep_data/datafiles/val_files.json','r'))
   # mode = 'train'
   # label_csv = '/import/research_c4dm/ss380/hyper_data/class_labels_indices.csv'
   # train_dataset = FSDataset(train_json,audio_conf,mode,label_csv)
   # train_dataset.__getitem__(0)







                








