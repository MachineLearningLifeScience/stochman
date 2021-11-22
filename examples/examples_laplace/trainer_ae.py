import os
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from data import get_data
from ae_models import get_encoder, get_decoder

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()

        latent_size = 2
        self.encoder = get_encoder(dataset, latent_size)
        self.decoder = get_decoder(dataset, latent_size)

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)    
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


def train_ae(dataset = "mnist"):

    # data
    train_loader, val_loader = get_data(dataset)

    # model
    model = LitAutoEncoder(dataset)

    # training
    n_device = torch.cuda.device_count()
    trainer = pl.Trainer(gpus=n_device, num_nodes=1, precision=16, limit_train_batches=0.5, max_epochs=100)
    trainer.fit(model, train_loader, val_loader)
    
    # save weights
    if not os.path.isdir(f"weights/{dataset}/"): os.makedirs(f"weights/{dataset}/")
    torch.save(model.encoder.state_dict(), f"weights/{dataset}/encoder.pth")
    torch.save(model.decoder.state_dict(), f"weights/{dataset}/decoder.pth")

if __name__ == "__main__":

    dataset = "mnist"

    # train or load auto encoder
    train_ae(dataset)