import librosa
import soundfile as sf
import h5py
import numpy as np
import json
import os
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
args = parser.parse_args()

common_root = '/data/EECS-MachineListeningLab/datasets'

paths = {
    'fsd50k': {
        'dev_meta_path': f'{common_root}/FSD50K/ground_truth/dev.csv',
        'eval_meta_path': f'{common_root}/FSD50K/ground_truth/eval.csv',
        'dev_audio_path': f'{common_root}/FSD50K/dev_audio',
        'eval_audio_path': f'{common_root}/FSD50K/eval_audio',
        'hdf_dir_dev':'/data/scratch/acw572/hdf_fsd50k/dev_audio',
        'hdf_dir_eval':'/data/scratch/acw572/hdf_fsd50k/eval_audio',
        'datafiles_dir':'/data/scratch/acw572/hdf_fsd50k/datafiles',
    },
    # Add other datasets here as needed
}
dataset = args.dataset



def resample_and_save_to_hdf5(file_path, hdf_dir,target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    os.makedirs(hdf_dir, exist_ok=True)
    hdf5_filename = os.path.basename(file_path).replace('.wav', '.h5')
    hdf5_path = os.path.join(hdf_dir, hdf5_filename)
    print(f'Saving to {hdf5_path}')
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('audio', data=audio.astype('float32'), compression="gzip")
    
    return hdf5_path

if dataset == 'fsd50k':
    dev_csv_path = paths[dataset]['dev_meta_path']
    eval_csv_path = paths[dataset]['eval_meta_path']
    dev_audio_path = paths[dataset]['dev_audio_path']
    eval_audio_path = paths[dataset]['eval_audio_path']
    hdf_dir_dev = paths[dataset]['hdf_dir_dev']
    hdf_dir_eval = paths[dataset]['hdf_dir_eval']
    datafiles_dir = paths[dataset]['datafiles_dir']
    dev_cnt = 0
    val_cnt = 0
    eval_cnt = 0
    fsd_tr_data = []
    fsd_val_data = []
    tr_meta = np.loadtxt(dev_csv_path, skiprows=1,dtype=str)
    eval_meta = np.loadtxt(eval_csv_path, skiprows=1,dtype=str)

    for i in range(len(tr_meta)):
        try:
            fileid = tr_meta[i].split(',"')[0]
            labels = tr_meta[i].split(',"')[2][0:-1]
            
            set_info = labels.split('",')[1]
        except:
            fileid = tr_meta[i].split(',')[0]
            labels = tr_meta[i].split(',')[2]
            set_info = tr_meta[i].split(',')[3][0:-1]
        hdf5_path = resample_and_save_to_hdf5(os.path.join(dev_audio_path, fileid + '.wav'),hdf_dir_dev,target_sr=16000)
        labels = labels.split('",')[0]
    
        label_list = labels.split(',')
    
        new_label_list = []

        for label in label_list:
            new_label_list.append(label)
    
        new_label_list = ','.join(new_label_list)
        cur_dict = {'wav': hdf5_path, 'labels': new_label_list}
        if set_info == 'trai':
            fsd_tr_data.append(cur_dict)
            dev_cnt += 1
        elif set_info == 'va':
            fsd_val_data.append(cur_dict)
            val_cnt += 1
        else:
            raise ValueError('unrecognized set')


    if not os.path.exists(datafiles_dir):
        os.makedirs(datafiles_dir)
    with open(os.path.join(datafiles_dir, 'fsd_tr_full.json'), 'w') as file:
        json.dump({"data": fsd_tr_data}, file, indent=1)
    with open(os.path.join(datafiles_dir, 'fsd_val_full.json'), 'w') as file:
        json.dump({"data": fsd_val_data}, file, indent=1)
            
    print('Processed {:d} samples for the FSD50K training set.'.format(dev_cnt))
    print('Processed {:d} samples for the FSD50K validation set.'.format(val_cnt))

    eval_cnt = 0

    fsd_eval_data = []

    for i in range(len(eval_meta)):
        try:
            fileid = eval_meta[i].split(',"')[0]
            labels = eval_meta[i].split(',"')[2][0:-1]
        except:
            fileid = eval_meta[i].split(',')[0]
            labels = eval_meta[i].split(',')[2]
        
        hdf5_path = resample_and_save_to_hdf5(os.path.join(eval_audio_path, fileid + '.wav'),hdf_dir_eval,target_sr=16000)
        labels = labels.split('",')[0]
    
        label_list = labels.split(',')
    
        new_label_list = []

        for label in label_list:
            new_label_list.append(label)
    
        new_label_list = ','.join(new_label_list)
        cur_dict = {'wav': hdf5_path, 'labels': new_label_list}
        fsd_eval_data.append(cur_dict)
        eval_cnt += 1
    with open(os.path.join(datafiles_dir, 'fsd_eval_full.json'), 'w') as file:
        json.dump({"data": fsd_eval_data}, file, indent=1)
    print('Processed {:d} samples for the FSD50K evaluation set.'.format(eval_cnt))






