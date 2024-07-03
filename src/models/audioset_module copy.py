from typing import Any, Dict, Tuple
import numpy as np 
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utilities.stats import calculate_stats
import wandb
import bisect
import torch.distributed as dist

class TaggingModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        loss:str,
        opt_warmup:bool,
        learning_rate:float,
        lr_rate:list,
        lr_scheduler_epoch:list,
        
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        self.net = net
        self.optimizer = optimizer
        self.warmup = opt_warmup
        self.scheduler = scheduler
        self.compile = compile
        self.loss = loss
        self.lr_scheduler_epoch = lr_scheduler_epoch
        self.lr_rate = lr_rate
        if self.loss == 'bce':
            self.criterion = torch.nn.BCELoss()
        elif self.loss == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_mAP = MeanMetric()
        self.test_mAP = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_mAP_best = MaxMetric()
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        self.milestones = [10,15,20,25,30,35,40]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mAP.reset()
        self.test_mAP.reset()
        self.val_mAP_best.reset()
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        
        preds = self.forward(x)
        epsilon = 1e-7
        #preds = torch.clamp(preds, epsilon, 1. - epsilon)
        
        loss = self.criterion(preds, y)
        
        return loss, preds, y
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss,preds,y = self.model_step(batch)
        
        self.train_loss(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        return loss
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        #preds = torch.sigmoid(preds)
        target_cpu = targets
        preds_cpu = preds
        self.val_predictions.append(preds_cpu)
        self.val_targets.append(target_cpu)
        
        #stats = calculate_stats(preds_cpu, target_cpu)
        #print(stats['AP'])
        # update and log metrics
        self.val_loss(loss)
        #self.val_mAP(stats['AP'])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #self.log("val/mAP", self.val_mAP, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        #mAP = self.val_mAP.compute()  # get current val acc
        #self.val_mAP_best(mAP)  # update best so far val acc
        val_preds = torch.cat(self.val_predictions, dim=0)
        val_targets = torch.cat(self.val_targets, dim=0)
        
        stats = calculate_stats(val_preds.cpu().detach().numpy(), val_targets.cpu().detach().numpy())
        mAP = np.mean([stat['AP'] for stat in stats])
        print(mAP)
        acc = stats[0]['acc']
        print(val_preds.shape)
        #self.val_mAP_best(mAP)
        #self.log("val/mAP", mAP, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

        if torch.cuda.device_count() > 1:
            gather_pred = [torch.zeros_like(val_preds) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(val_targets) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(gather_pred, val_preds)
            dist.all_gather(gather_target, val_targets)
            
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim=0).cpu().detach().numpy()
                gather_target = torch.cat(gather_target, dim=0).cpu().detach().numpy()
                print(gather_pred.shape)
                stats = calculate_stats(gather_pred, gather_target)
                mAP = np.mean([stat['AP'] for stat in stats])
        #mAUC = np.mean(stats['AUC'])
                acc = stats[0]['acc']
                self.val_mAP_best(mAP)
                print("Logging on rank 0")
                print(f'mAP value {mAP}')
                self.log("val/mAP", mAP * float(dist.get_world_size()), on_step=False, on_epoch=True, prog_bar=True,rank_zero_only=True)
               # self.log("val/mAP_best", self.val_mAP_best.compute(),prog_bar=True,rank_zero_only=True)
            #dist.barrier()

            
        else:
            stats = calculate_stats(val_preds.cpu().detach().numpy(), val_targets.cpu().detach().numpy())
            mAP = np.mean([stat['AP'] for stat in stats])
            acc = stats[0]['acc']
            self.val_mAP_best(mAP)
            self.log("val/mAP", mAP, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        
        self.val_predictions.clear()
        self.val_targets.clear()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss, preds, targets = self.model_step(batch)
        
        
        preds_cpu = preds
        target_cpu = targets
        self.test_predictions.append(preds_cpu)
        self.test_targets.append(target_cpu)

        # update and log metrics
        self.test_loss(loss)
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("test/mAP", self.test_mAP, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        test_preds = torch.cat(self.test_predictions, dim=0)
        test_targets = torch.cat(self.test_targets, dim=0)
        if torch.cuda.device_count() > 1:
            gather_pred = [torch.zeros_like(test_preds) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(test_targets) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(gather_pred, test_preds)
            dist.all_gather(gather_target, test_targets)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim=0).cpu().detach().numpy()
                gather_target = torch.cat(gather_target, dim=0).cpu().detach().numpy()
                stats = calculate_stats(gather_pred, gather_target)
        
                mAP = np.mean([stat['AP'] for stat in stats])
        #mAUC = np.mean(stats['AUC'])
                acc = stats[0]['acc']
                print("Logging test metrics on rank 0")
                print(f'mAP value {mAP}')
                self.log("test/mAP", mAP, on_step=False, on_epoch=True, prog_bar=True,sync_dist=False)

        else:
                
                stats = calculate_stats(test_preds.cpu().detach().numpy(), test_targets.cpu().detach().numpy())
                mAP = np.mean([stat['AP'] for stat in stats])
                acc = stats[0]['acc']
                
                self.log("test/mAP", mAP * float(dist.get_world_size()), on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/mAP_best", self.val_mAP_best.compute(), sync_dist=True, prog_bar=True)
        self.test_predictions.clear()
        self.test_targets.clear()

    def setup(self, stage:str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
    

    def configure_optimizers(self) -> Dict[str, Any]:
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.

    #     Examples:
    #         https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

    #     :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
    #     """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        # if self.hparams.scheduler is not None:
        #     def lr_foo(epoch):
             
        #      if epoch < 1:
        #          # warm up lr
        #          lr_scale = self.lr_rate[epoch]
        #          print(f'warmup lr_scale:{lr_scale}')
        #      else:
        #          # warmup schedule
        #         lr_pos = int(-1 - bisect.bisect_left(self.lr_scheduler_epoch, epoch))
        #         if lr_pos < -3:
        #             lr_scale = max(self.lr_rate[0] * (0.98 ** epoch), 0.03)
        #             print(f'nonwarmup first lr_scale:{lr_scale}')
        #         else:
        #             lr_scale = self.lr_rate[lr_pos]
        #             lr_scale = 0.95 ** epoch
        #             print(f'nonwarmup second lr_scale:{lr_scale}')
        #      return lr_scale

        
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        

        # if self.hparams.scheduler is not None:
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(
        #      optimizer,
        #      lr_lambda=lr_foo)
        
        #     #scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        # return {"optimizer": optimizer}
        # if self.hparams.scheduler is not None:
            
        #     def lr_foo(epoch):
             
        #      if epoch < 1:
        #          # warm up lr
        #          lr_scale = self.lr_rate[epoch]
        #      else:
        #          # warmup schedule
        #         lr_pos = int(-1 - bisect.bisect_left(self.milestones, epoch))
        #         if lr_pos < -3:
        #             lr_scale = max(self.lr_rate[0] * (0.98 ** epoch), 0.03)
        #         else:
        #             lr_scale = self.lr_rate[lr_pos]
        #             lr_scale = 0.95 ** epoch
        #      return lr_scale
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lr_foo)
         
        #     return {
        #           "optimizer": optimizer,
        #           "lr_scheduler": {
        #               "scheduler": scheduler,
        #               "monitor": "val/loss",
        #               "interval": "epoch",
        #               "frequency": 1,
        #           },}
        # return {"optimizer": optimizer}
         
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        # return {"optimizer": optimizer}
    
        # def lr_foo(epoch):
        #     if epoch < 1:
        #         # warm up lr
        #         lr_scale = self.lr_rate[epoch]
        #     else:
        #         # warmup schedule
        #         #lr_pos = int(-1 - bisect.bisect_left(self.milestones, epoch))
        #         #if lr_pos < -3:
        #         #    lr_scale = max(self.lr_rate[0] * (0.98 ** epoch), 0.03)
        #         #else:
        #         #    lr_scale = self.lr_rate[lr_pos]
        #         lr_scale = 0.95 ** epoch
        #     return lr_scale
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lr_foo
        # )
        # return {
        #          "optimizer": optimizer,
        #          "lr_scheduler": {
        #              "scheduler": scheduler,
        #              "monitor": "val/loss",
        #              "interval": "epoch",
        #              "frequency": 1,
        #          },
        # }
        #  if self.hparams.scheduler is None:
        #     print("No scheduler")
        #  #print(optimizer)
        #  if self.hparams.scheduler is not None:
             
             #scheduler = self.hparams.scheduler(optimizer=optimizer)
             
            # }
         #return {"optimizer": optimizer}
         #return torch.optim.Adam(self.net.parameters(), lr=5e-4)
    

    # def optimizer_step(self,
    #                     epoch,
    #                     batch_idx,
    #                     optimizer,
    #                     optimizer_closure,
    #   ):
        
        
    #      if self.trainer.global_step <= 1000 and self.trainer.global_step % 50 == 0 and self.warmup == True:
            
    #          warm_lr = (self.trainer.global_step / 1000) * optimizer.param_groups[0]['lr']
    #          for pg in optimizer.param_groups:
    #              pg['lr'] = warm_lr
        
    #      optimizer.step(closure=optimizer_closure)
    
            


        

        
   
    



    





    







    

    

    

    
