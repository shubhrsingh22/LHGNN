import json
import os
import pandas 
import numpy as np 
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple
import hydra 
import rootutils

#rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def prep_json(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    print("Preparing JSON files for {dataset}")
    print(cfg.path)
    #print(cfg.path)

    














@hydra.main(version_base="1.3",config_path="../configs", config_name="prep_data.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print("In main")
    prep_data(cfg)


if __name__ == "__main__":
    main()