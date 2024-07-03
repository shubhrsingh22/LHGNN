from typing import Any, Dict, List, Optional, Tuple

import hydra
#import lightning as L
import pytorch_lightning as pl
import rootutils
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,Callback
from pytorch_lightning.loggers import Logger
#from lightning import Callback, LightningDataModule, LightningModule, Trainer
#from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
from lightning.pytorch.tuner import Tuner
print(sys.path)


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True,cwd=False)
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ["WANDB__SERVICE_WAIT"] = "300"
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    state_dict = torch.load(cfg.get('ckpt_path'))['state_dict']
    
    

    own_state = model.net.state_dict()
    
    for name, param in state_dict.items():
        
        
        if name == 'stem.convs.0.weight':
            
            param = torch.mean(param,dim=1,keepdim=True)
        
        if 'pos_embed' in name:
            
            param = F.interpolate(param, size=(cfg.data.get('num_mels') // 4,cfg.data.get('target_len') // 4), mode='bicubic', align_corners=False)
                        
        if 'relative_pos' in name:

            target_shape = own_state[name].shape[-2:]
            h = own_state[name].shape[1]
            w = own_state[name].shape[2]
            param = F.interpolate(param.unsqueeze(1), size=(h,w), mode='bicubic', align_corners=False).squeeze(1)
        
        if 'prediction' in name:
            continue

        own_state[name].copy_(param)
    
    model.net.load_state_dict(own_state)
    
   # model.net._modules['prediction'][-1] = nn.Conv2d(1024,cfg.model.net.get('num_class'),1,bias=True)




     
    # print(state_dict.keys())
    # Add 'net.' prefix to every key
   # model.net.load_state_dict(torch.load(cfg.get('ckpt_path')),strict=True)
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    #tuner = Tuner(trainer)
    
    #lr_finder = tuner.lr_find(model,datamodule)
    #new_lr = lr_finder.suggestion()
    #print(new_lr)
    #model.hyparams.lr = new_lr
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "callbacks": callbacks,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")

        log.info("Finding optimal learning rate!")
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics
    
    

        
    if cfg.get("test"):
        log.info("Starting testing for single model!")
        ckpt_path = trainer.checkpoint_callback.best_model_path

        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    if cfg.get('wa'):
            
        log.info("Weighted average model")
        model_ckpt = []
        ckpt_dir = cfg.callbacks.get('model_checkpoint').get('dirpath')
        
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        own_state = model.state_dict()
        for ckpt in os.listdir(ckpt_dir):

            if 'ckpt' in ckpt:
                #print(torch.load(os.path.join(ckpt_dir,ckpt))['state_dict'].keys())
                model_ckpt.append(torch.load(os.path.join(ckpt_dir,ckpt))['state_dict'])
        
        
        for name, params in own_state.items():
            own_state[name] = torch.zeros_like(params)
            model_ckpt_key = torch.cat([d[name].float().unsqueeze(0) for d in model_ckpt],dim=0)
            own_state[name].copy_(torch.mean(model_ckpt_key,dim=0))
        
        model.load_state_dict(own_state)
        log.info("Starting testing for weighted average model!")
        test_results_wa = trainer.test(model=model, datamodule=datamodule)
        torch.save(model.net.state_dict(),os.path.join(ckpt_dir,'wa.pth.tar'))
        log.info(f"Weighted average results : {test_results['test/mAP']}")
        logger.log('weighted test/mAP',test_results_wa['test/mAP'])

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    #metric_dict = {**train_metrics, **test_metrics}

    #return metric_dict, object_dict
    return train_metrics,object_dict

    
        


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    train(cfg)
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    #metric_value = get_metric_value(
     #   metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    #)

    # return optimized metric
   # return metric_value


if __name__ == "__main__":
    print("In main")
    torch.set_float32_matmul_precision("high")
    main()
