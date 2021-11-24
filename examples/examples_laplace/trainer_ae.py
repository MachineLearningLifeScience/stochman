import os
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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


def test_ae(dataset, batch_size=1):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    latent_size = 2
    encoder = get_encoder(dataset, latent_size).eval().to(device)
    encoder.load_state_dict(torch.load(f"weights/{dataset}/encoder.pth"))

    decoder = get_decoder(dataset, latent_size).eval().to(device)
    decoder.load_state_dict(torch.load(f"weights/{dataset}/decoder.pth"))

    train_loader, val_loader = get_data(dataset, batch_size)

    # forward eval
    x, z, x_rec, labels = [], [], [], [] 
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.view(xi.size(0), -1).to(device)
        with torch.inference_mode():
            zi = encoder(xi)
            x_reci = decoder(zi)

            x += [xi.cpu()]
            z += [zi.detach().cpu()]
            x_rec += [x_reci.detach().cpu()]
            labels += [yi]

        # only show the first 50 points
        # if i > 50:
        #    break
    
    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z = torch.cat(z, dim=0).numpy()
    x_rec = torch.cat(x_rec, dim=0).numpy()

    # create figures
    if not os.path.isdir(f"figures/{dataset}"): os.makedirs(f"figures/{dataset}")

    plt.figure()
    if dataset == "mnist":
        for yi in np.unique(labels):
            idx = labels == yi
            plt.plot(z[idx, 0], z[idx, 1], 'x', ms=5.0, alpha=1.0)
    else:
        plt.plot(z[:, 0], z[:, 1], 'x', ms=5.0, alpha=1.0)

    plt.savefig(f"figures/{dataset}/ae_contour.png")
    plt.close(); plt.cla()

    if dataset == "mnist":
        for i in range(min(len(z), 10)):
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(x[i].reshape(28,28))

            plt.subplot(1,2,2)
            plt.imshow(x_rec[i].reshape(28,28))

            plt.savefig(f"figures/{dataset}/ae_recon_{i}.png")
            plt.close(); plt.cla()


def train_ae(dataset = "mnist"):

    # data
    train_loader, val_loader = get_data(dataset)

    # model
    model = LitAutoEncoder(dataset)

    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    # early stopping
    callbacks = [EarlyStopping(monitor="val_loss")]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_device, num_nodes=1, auto_scale_batch_size=True, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    
    # save weights
    if not os.path.isdir(f"weights/{dataset}/"): os.makedirs(f"weights/{dataset}/")
    torch.save(model.encoder.state_dict(), f"weights/{dataset}/encoder.pth")
    torch.save(model.decoder.state_dict(), f"weights/{dataset}/decoder.pth")

if __name__ == "__main__":

    dataset = "mnist"
    train = True

    # train or load auto encoder
    if train:
        train_ae(dataset)

    test_ae(dataset)

    