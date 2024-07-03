import numpy as np
import json
import os

import numpy as np
import json
import os
import csv


path = '/homes/ss380/deeplearn/graphtrain/hypergraph/hypergraph_exps/HGANN/datafiles/custom.eval_segments.csv'
root_path = '/data/EECS-MachineListeningLab/datasets/AudioSet/audios'
fsd_tr_data = []

with open(path, 'r') as file:
  csvreader = csv.reader(file,delimiter='|')
  ix = 0
  for row in csvreader:
    if not ix == 0:
      file_path = os.path.join(root_path, row[0])
      print(file_path)
      label_list = row[1].split(',')
      new_label_list = []
      for label in label_list:
        new_label_list.append(label)
      new_label_list = ','.join(new_label_list)
      
      
      cur_dict = {'wav': file_path, 'labels': new_label_list}
      fsd_tr_data.append(cur_dict)
    ix += 1
  print(f'total validation files: {ix}')
  with open('/homes/ss380/deeplearn/graphtrain/hypergraph/hypergraph_exps/HGANN/datafiles/eval_full.json', 'w') as f:
          json.dump({'data': fsd_tr_data}, f, indent=1)

      
    
      
    
    

#train_segments = json.load(open('/homes/ss380/deeplearn/graphtrain/hypergraph/hypergraph_exps/HGANN/datafiles/custom.all_train_segments.json'))

#label_list = train_segments[3]['machine_id']
#print(type(label_list))