import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Finetuner(pl.LightningModule):
    def __init__(self, model, optimizer, inp_scale=False, bounds=None, max_epochs=1000):
        super(Finetuner, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.inp_scale = inp_scale
        if bounds is not None:
            self.lb, self.ub = bounds

    def forward(self, x):
        H = x
        if self.inp_scale: H = self.neural_net_scale(x)
        return self.model(H)

    def neural_net_scale(self, inp):
        return 2*(inp-self.lb)/(self.ub-self.lb)-1

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
