import librosa
import soundfile as sf
import h5py
import numpy as np
import json
import os
import csv
import argparse




common_root = '/data/EECS-MachineListeningLab/datasets/AudioSet/'
bal_tr_meta = os.path.join(common_root, "meta/custom.balanced_train_segments.csv")
all_tr_meta = os.path.join(common_root, "meta/custom.all_train_segments.csv")
eval_meta = os.path.join(common_root, "meta/custom.eval_segments.csv")
bal_tr_meta = np.loadtxt(bal_tr_meta, skiprows=1,dtype=str)
all_tr_meta = np.loadtxt(all_tr_meta, skiprows=1,dtype=str)
eval_meta = np.loadtxt(eval_meta, skiprows=1,dtype=str)
bal_audio_path = os.path.join(common_root,'audios') 
unbal_audio_path = os.path.join(common_root,'audios')
eval_audio_path = os.path.join(common_root,'audios')
hdf_dir_bal = '/data/scratch/acw572/AudioSet/audios/balanced_segments'
hdf_dir_all = '/data/scratch/acw572/AudioSet/audios/unbalanced_segments'
hdf_dir_eval = '/data/scratch/acw572/AudioSet/audios/eval_segments'
datafiles_dir = '/data/scratch/acw572/AudioSet/datafiles'






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

bal_cnt,eval_cnt,all_tr_cnt = 0,0,0
bal_tr_data = []
eval_data = []
all_tr_data = []

# for i in range(len(bal_tr_meta)):
#     try:
#         fileid = bal_tr_meta[i].split('|')[0]
#         labels = bal_tr_meta[i].split('|')[1]
#     except:
#         fileid = bal_tr_meta[i].split('.')[0]
#         labels = bal_tr_meta[i].split('|')[1]
#     print(fileid)
#     hdf5_path = resample_and_save_to_hdf5(os.path.join(bal_audio_path, fileid),hdf_dir_bal,target_sr=16000)
#     labels = labels.split('",')[0]
    
#     label_list = labels.split(',')
    
#     new_label_list = []

#     for label in label_list:
#         new_label_list.append(label)
    
#     new_label_list = ','.join(new_label_list)
#     cur_dict = {'wav': hdf5_path, 'labels': new_label_list}
#     if len(new_label_list) == 0:
#         continue
#     if len(new_label_list) == 1:
#         bal_tr_data.append([fileid,new_label_list[0]])
#         bal_cnt += 1
#     else:
#         bal_tr_data.append([fileid,new_label_list])
#         bal_cnt += 1

# if not os.path.exists(datafiles_dir):
#         os.makedirs(datafiles_dir)
# with open(os.path.join(datafiles_dir, 'balanced_train_segments.json'), 'w') as f:
#     json.dump({"data": bal_tr_data}, f, indent=1)

# print('Processed {:d} samples for the Audioset balanced training set.'.format(bal_cnt))


for i in range(len(all_tr_meta)):
    try:
        fileid = all_tr_meta[i].split('|')[0]
        labels = all_tr_meta[i].split('|')[1]
    except:
        fileid = all_tr_meta[i].split('.')[0]
        labels = all_tr_meta[i].split('|')[1]
    hdf5_path = resample_and_save_to_hdf5(os.path.join(unbal_audio_path, fileid),hdf_dir_all,target_sr=16000)
    labels = labels.split('",')[0]

    label_list = labels.split(',')

    new_label_list = []
    for label in label_list:
        new_label_list.append(label)
    new_label_list = ','.join(new_label_list)
    cur_dict = {'wav': hdf5_path, 'labels': new_label_list}
    if len(new_label_list) == 0:
        continue
    if len(new_label_list) == 1:
        all_tr_data.append([fileid,new_label_list[0]])
        all_tr_cnt += 1
    else:
        all_tr_data.append([fileid,new_label_list])
        all_tr_cnt += 1
    if all_tr_cnt % 10000 == 0:
        print('Processed {:d} samples for the Audioset unbalanced training set.'.format(all_tr_cnt))
with open(os.path.join(datafiles_dir, 'unbalanced_train_segments.json'), 'w') as f:
    json.dump({"data": all_tr_data}, f, indent=1)
print('Processed {:d} samples for the Audioset unbalanced training set.'.format(all_tr_cnt))

for i in range(len(eval_meta)):
    try:
        fileid = eval_meta[i].split(',"')[0]
        labels = eval_meta[i].split(',"')[2][0:-1]
    except:
        fileid = eval_meta[i].split(',')[0]
        labels = eval_meta[i].split(',')[2]
    hdf5_path = resample_and_save_to_hdf5(os.path.join(eval_audio_path, fileid),hdf_dir_eval,target_sr=16000)
    labels = labels.split('",')[0]
    label_list = labels.split(',')
    new_label_list = []
    for label in label_list:
        new_label_list.append(label)
    label_list = ','.join(new_label_list)
    cur_dict = {'wav': hdf5_path, 'labels': label_list}
    if len(label_list) == 0:
        continue
    if len(label_list) == 1:
        eval_data.append([fileid,label_list[0]])
        eval_cnt += 1
    else:
        eval_data.append([fileid,label_list])
        eval_cnt += 1
with open(os.path.join(datafiles_dir, 'eval_segments.json'), 'w') as f:
    json.dump({"data": eval_data}, f, indent=1)
print('Processed {:d} samples for the Audioset eval set.'.format(eval_cnt))
