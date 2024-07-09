import numpy as np 
import json 
import os 


root_path = "/data/EECS-MachineListeningLab/datasets/FSD50K"
tr_meta = os.path.join(root_path,"ground_truth/dev.csv")
tr_meta = os.path.join(root_path,"ground_truth/dev.csv")

tr_meta = np.loadtxt(tr_meta, skiprows=1,dtype=str)
eval_meta = os.path.join(root_path, "ground_truth/eval.csv")
eval_meta = np.loadtxt(eval_meta, skiprows=1,dtype=str)
wrk_path = "/data/home/acw572/hgann/HGANN"

tr_cnt, val_cnt = 0, 0

fsd_tr_data = []
fsd_val_data = []
fsd_eval_data = []

for i in range(len(tr_meta)):
    
    try:
        fileid = tr_meta[i].split(',"')[0]
        labels = tr_meta[i].split(',"')[2][0:-1]
        print(labels)
        set_info = labels.split('",')[1]
    except:
        fileid = tr_meta[i].split(',')[0]
        labels = tr_meta[i].split(',')[2]
        set_info = tr_meta[i].split(',')[3][0:-1]
    
    
    labels = labels.split('",')[0]
    
    label_list = labels.split(',')
    
    new_label_list = []
    for label in label_list:
        new_label_list.append(label)
    
    new_label_list = ','.join(new_label_list)
    
    
    cur_dict = {"wav": root_path + '/dev_audio/' + fileid + '.wav', 
                "labels": new_label_list}
    
    if set_info == 'trai':
        fsd_tr_data.append(cur_dict)
        tr_cnt += 1
    elif set_info == 'va':
        fsd_val_data.append(cur_dict)
        val_cnt += 1
    else:
        raise ValueError('unrecognized set')
    

    
datafile_path = os.path.join(wrk_path,'datafiles')
tr_json = os.path.join(datafile_path, 'fsd_tr_full.json')
val_json = os.path.join(datafile_path, 'fsd_val_full.json')
eval_json = os.path.join(datafile_path, 'fsd_eval_full.json')


if not os.path.exists(datafile_path):
    os.makedirs(datafile_path)

with open(tr_json, 'w') as file:
    json.dump({"data": fsd_tr_data}, file, indent=1)

with open(val_json, 'w') as file:
    json.dump({"data": fsd_val_data}, file, indent=1)

cnt = 0 

for i in range(len(eval_meta)):
    try:
        fileid = eval_meta[i].split(',"')[0]
        labels = eval_meta[i].split(',"')[2][0:-1]
        
    except:
        fileid = eval_meta[i].split(',')[0]
        labels = eval_meta[i].split(',')[2]
        
    
    label_list = labels.split(',')
    new_label_list = []
    for label in label_list:
        new_label_list.append(label)
    
    if len(new_label_list) != 0:
        new_label_list = ','.join(new_label_list)
        cur_dict = {"wav": root_path + '/eval_audio/' + fileid + '.wav', "labels":new_label_list}
        fsd_eval_data.append(cur_dict)
        cnt +=1
    
with open(eval_json, 'w') as file:
    json.dump({"data": fsd_eval_data}, file, indent=1)
print('Processed {:d} samples for the FSD50K evaluation set.'.format(cnt))
print('Processed {:d} samples for the FSD50K training set.'.format(tr_cnt))
print('Processed {:d} samples for the FSD50K validation set.'.format(val_cnt))


    



