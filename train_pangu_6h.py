# AIWx

import lightning as L
import torch, torch.utils.data as data
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from src.trainer.iter_pangu import Trainer
from src.dataset.dataset_pangu import Dataset_pangu

# Fix random Seed and setup
L.seed_everything(727, workers=True)
torch.set_float32_matmul_precision('medium')
wandb_logger = WandbLogger(project="AIWF",name="Pangu-weather 6h",offline=True)

# save best model
checkpoint_callback=ModelCheckpoint(monitor="val/rmse_score", save_top_k=3, dirpath='bestmodel/pangu_6h', filename='epoch{epoch:02d}-score{val/rmse_score:.2f}',auto_insert_metric_name=False)

# Dataset
train = Dataset_pangu(split='train')
val = Dataset_pangu(split='val')

trainer = L.Trainer(max_epochs=50, strategy=DDPStrategy(find_unused_parameters=False),logger=wandb_logger, precision="bf16-mixed", check_val_every_n_epoch=1, callbacks=[checkpoint_callback])

model=Trainer(iter_per_epoch=731,
                    eta_max=5e-4)

trainer.fit(model, data.DataLoader(train, batch_size=1,num_workers=2,shuffle=True,drop_last=True),data.DataLoader(val, batch_size=2,num_workers=2))
