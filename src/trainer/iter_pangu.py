# AIWx
# We conducted the reproduction based on the pseudo-code provided by the official website 
# https://github.com/198808xc/Pangu-Weather
# <Accurate medium-range global weather forecasting with 3D neural networks>

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 2)[0]  # 上三级目录
sys.path.append(config_path)

import math
 
import torch
import lightning as L

from src.model.pangu.py import ForeCastModel
from src.tools.metrics import compute_latitude_weighted_rmse
from src.loss.loss_pangu import latitude_and_variable_weighted_mae

climates = ['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50', 
            'q1000', 'q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50',
            't1000', 't925', 't850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50',
            'u1000', 'u925', 'u850', 'u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50',
            'v1000', 'v925', 'v850', 'v700', 'v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50',
            'msl', 'u10', 'v10', 't2m']

class Trainer(L.LightningModule):
    def __init__(self,
                 iter_per_epoch=914,
                 eta_min=1e-7,
                 eta_max=5e-4,
                 warmup_rate=0.1,
                 weight_decay=3e-6):
        
        super().__init__()
        self.model=ForeCastModel()
        
        self.validation_step_outputs = []
        self.samples = 0
        self.iter_per_epoch = iter_per_epoch
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.warmup_rate = warmup_rate
        self.weight_decay = weight_decay
    
    def training_step(self, batch, batch_idx):
        (input,input_surface), (target,target_surface)=batch
        
        output,output_surface = self.model(input,input_surface)
        loss = latitude_and_variable_weighted_mae(target,target_surface,output,output_surface)

        self.log("train/lr",self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch):
        pred=[]
        (input, input_surface), target = batch
        with torch.no_grad():
            for _ in range(20):
                output,output_surface  = self.model(input,input_surface) 
                pred_air = output
                pred_surface = output_surface
                pred.append(torch.concat([pred_air,pred_surface],dim=2))

                input = output
                input_surface = output_surface
                input=input.detach()
                input_surface=input_surface.detach()

        pred=torch.concatenate(pred,dim=1)
        for cid, _ in enumerate(climates):
            self.validation_step_outputs.append([])
            for sid in range(pred.shape[1]):
                out = pred[:, sid, cid]
                tgt = target[:, sid, cid]
                rmse = self.compute_latitude_weighted_rmse(out, tgt)
                self.validation_step_outputs[-1].append(rmse)
        
        self.samples += input.shape[0]

    def on_validation_epoch_end(self):
        result = []
        for cid, name in enumerate(climates):
            flatten=[]
            for row in self.validation_step_outputs[cid::69]:
                flatten+=row
            score =torch.sum(torch.stack(flatten))/(self.samples*20)
            result.append(score)
            # Average to the main thread.
            self.log("val/rmse_{}".format(name),score,sync_dist=True)

        self.log("val/rmse_score",torch.mean(torch.stack(result)),sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        self.samples = 0
        
    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # warm-up cosanneal Lr
        T_max = self.iter_per_epoch * self.trainer.max_epochs
        warm_up_iter = T_max * self.warmup_rate

        eta_max = self.eta_max
        eta_min = self.eta_min

        optimizer=torch.optim.AdamW(
            [{'params': self.model.parameters(), 'lr':eta_max}],
            weight_decay=self.weight_decay, 
            betas=(0.9, 0.95)
            )
        
        lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else (eta_min + 0.5*(eta_max-eta_min)*(1.0+math.cos((cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/eta_max
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    'scheduler': scheduler,
                    "interval": "step",},
                }