import numpy as np
import json
import os

import numpy as np
import json
import os

json_path = '/homes/ss380/deeplearn/graphtrain/hypergraph/hypergraph_exps/HGANN/datafiles/eval_full.json'
bask_path = '/bask/projects/v/vjgo8416-lrg-audio/datasets/AudioSet/audios'
with open(json_path, 'r') as file:
  data_json = json.load(file)
  counter = 0
  for item in data_json['data']:
     
    item['wav'] = item['wav'].replace('/data/EECS-MachineListeningLab/datasets/AudioSet/audios/eval_segments/', '/bask/projects/v/vjgo8416-lrg-audio/datasets/AudioSet/audios/eval_segments/')

with open(json_path, 'w') as file:
  json.dump(data_json, file, indent=1)
  



    

    
    
    
    

        
       
       
    

  