import os
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import get_data, generate_latent_grid
from ae_models import get_encoder, get_decoder


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


class LitDropoutAutoEncoder(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()

        latent_size = 2
        self.encoder = get_encoder(dataset, latent_size, dropout=True)
        self.decoder = get_decoder(dataset, latent_size, dropout=True)

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

        # activate dropout layers
        apply_dropout(self.encoder)
        apply_dropout(self.decoder)

        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


def test_mcdropout_ae(dataset, batch_size=1):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    latent_size = 2
    encoder = get_encoder(dataset, latent_size, dropout=True).eval().to(device)
    encoder.load_state_dict(torch.load(f"weights/{dataset}/mcdropout_ae/encoder.pth"))

    decoder = get_decoder(dataset, latent_size, dropout=True).eval().to(device)
    decoder.load_state_dict(torch.load(f"weights/{dataset}/mcdropout_ae/decoder.pth"))

    train_loader, val_loader = get_data(dataset, batch_size)

    # number of mc samples
    N = 30

    # forward eval
    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = [], [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.view(xi.size(0), -1).to(device)
        with torch.inference_mode():

            # activate dropout layers
            apply_dropout(encoder)
            apply_dropout(decoder)

            mu_z_i, mu2_z_i, mu_rec_i, mu2_rec_i = None, None, None, None
            for n in range(N):

                zi = encoder(xi)
                x_reci = decoder(zi)

                # compute running mean and running variance
                if mu_rec_i is None:
                    mu_rec_i = x_reci
                    mu2_rec_i = x_reci**2
                    mu_z_i = zi
                    mu2_z_i = zi**2
                else:
                    mu_rec_i += x_reci
                    mu2_rec_i += x_reci**2
                    mu_z_i += zi
                    mu2_z_i += zi**2

            mu_rec_i = mu_rec_i / N
            mu2_rec_i = mu2_rec_i / N

            # add abs for numerical stability
            sigma_rec_i = (mu2_rec_i - mu_rec_i**2).abs().sqrt()

            mu_z_i = mu_z_i / N
            mu2_z_i = mu2_z_i / N

            # add abs for numerical stability
            sigma_z_i = (mu2_z_i - mu_z_i**2).abs().sqrt()

            x += [xi.cpu()]
            z_mu += [mu_z_i.detach().cpu()]
            z_sigma += [sigma_z_i.detach().cpu()]
            x_rec_mu += [mu_rec_i.detach().cpu()]
            x_rec_sigma += [sigma_rec_i.detach().cpu()]
            labels += [yi]

        # only show the first 50 points
        # if i > 50:
        #    break
    
    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.cat(z_mu, dim=0).numpy()
    z_sigma = torch.cat(z_sigma, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    x_rec_sigma = torch.cat(x_rec_sigma, dim=0).numpy()

    # Grid for probability map
    z_grid_loader = generate_latent_grid(
        z_mu[:, 0].min(),
        z_mu[:, 0].max(),
        z_mu[:, 1].min(),
        z_mu[:, 1].max()
    )
    
    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):
        
        z_grid = z_grid[0].to(device)

        mu_rec_grid, mu2_rec_grid = None, None
        with torch.inference_mode():
            
            # enable dropout
            apply_dropout(decoder)

            # take N mc samples
            for n in range(N):

                x_recn = decoder(z_grid)

                # compute running mean and variance
                if mu_rec_grid is None:
                    mu_rec_grid = x_recn
                    mu2_rec_grid = x_recn**2
                else:
                    mu_rec_grid += x_recn
                    mu2_rec_grid += x_recn**2

        mu_rec_grid = mu_rec_grid / N
        mu2_rec_grid = mu2_rec_grid / N

        # add abs for numerical stability
        sigma_rec_grid = (mu2_rec_grid - mu_rec_grid**2).abs().sqrt()

        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [sigma_rec_grid.cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    sigma_vector = f_sigma.mean(axis=1)

    # create figures
    if not os.path.isdir(f"figures/{dataset}/mcdropout_ae/"): os.makedirs(f"figures/{dataset}/mcdropout_ae/")

    plt.figure()
    if dataset == "mnist":
        for yi in np.unique(labels):
            idx = labels == yi
            plt.plot(z_mu[idx, 0], z_mu[idx, 1], 'x', ms=5.0, alpha=1.0)
    else:
        plt.plot(z_mu[:, 0], z_mu[:, 1], 'x', ms=5.0, alpha=1.0)

    precision_grid = np.reshape(sigma_vector, (n_points_axis, n_points_axis))
    plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
    plt.colorbar()

    plt.savefig(f"figures/{dataset}/mcdropout_ae/mcdropout_ae_contour.png")
    plt.close(); plt.cla()

    if dataset == "mnist":
        for i in range(min(len(z_mu), 10)):
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(x[i].reshape(28,28))

            plt.subplot(1,3,2)
            plt.imshow(x_rec_mu[i].reshape(28,28))

            plt.subplot(1,3,3)
            plt.imshow(x_rec_sigma[i].reshape(28,28))

            plt.savefig(f"figures/{dataset}/mcdropout_ae/mcdropout_ae_recon_{i}.png")
            plt.close(); plt.cla()


def train_mcdropout_ae(dataset = "mnist"):

    # data
    train_loader, val_loader = get_data(dataset)

    # model
    model = LitDropoutAutoEncoder(dataset)

    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    # early stopping
    callbacks = [EarlyStopping(monitor="val_loss")]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_device, num_nodes=1, auto_scale_batch_size=True, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    
    # save weights
    if not os.path.isdir(f"weights/{dataset}/mcdropout_ae/"): os.makedirs(f"weights/{dataset}/mcdropout_ae/")
    torch.save(model.encoder.state_dict(), f"weights/{dataset}/mcdropout_ae/encoder.pth")
    torch.save(model.decoder.state_dict(), f"weights/{dataset}/mcdropout_ae/decoder.pth")

if __name__ == "__main__":

    dataset = "mnist"
    train = False

    # train or load auto encoder
    if train:
        train_mcdropout_ae(dataset)

    test_mcdropout_ae(dataset)

    