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

from data import get_data, generate_latent_grid
from ae_models import get_encoder, get_decoder
from utils import softclip


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, dataset, use_var_decoder):
        super().__init__()

        self.use_var_decoder = use_var_decoder

        latent_size = 2
        self.encoder = get_encoder(dataset, latent_size)
        self.mu_decoder = get_decoder(dataset, latent_size)
        if self.use_var_decoder:
            self.var_decoder = get_decoder(dataset, latent_size)

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
        mu_x_hat = self.mu_decoder(z)
        
        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3)
            
            # reconstruction term: 
            loss = (torch.pow((mu_x_hat - x) / torch.exp(log_sigma_x_hat), 2) + log_sigma_x_hat).mean()
        else:
            loss = F.mse_loss(mu_x_hat, x)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        mu_x_hat = self.mu_decoder(z)
        
        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3)

            # reconstruction term: 
            loss = (torch.pow((mu_x_hat - x) / torch.exp(log_sigma_x_hat), 2) + log_sigma_x_hat).mean()
        else:
            loss = F.mse_loss(mu_x_hat, x)

        self.log('val_loss', loss)


def test_ae(dataset, batch_size=1, use_var_decoder=False):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{dataset}/ae_[use_var_dec={use_var_decoder}]"
    
    latent_size = 2
    encoder = get_encoder(dataset, latent_size).eval().to(device)
    encoder.load_state_dict(torch.load(f"weights/{path}/encoder.pth"))

    mu_decoder = get_decoder(dataset, latent_size).eval().to(device)
    mu_decoder.load_state_dict(torch.load(f"weights/{path}/mu_decoder.pth"))

    if use_var_decoder:
        var_decoder = get_decoder(dataset, latent_size).eval().to(device)
        var_decoder.load_state_dict(torch.load(f"weights/{path}/var_decoder.pth"))

    train_loader, val_loader = get_data(dataset, batch_size)

    # forward eval
    x, z, x_rec_mu, x_rec_sigma, labels = [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.view(xi.size(0), -1).to(device)
        with torch.inference_mode():
            zi = encoder(xi)
            x_reci = mu_decoder(zi)

            x += [xi.cpu()]
            z += [zi.cpu()]
            x_rec_mu += [x_reci.cpu()]
            labels += [yi]

            if use_var_decoder:
                x_rec_sigma += [softclip(var_decoder(zi), min=-6).cpu()]

        # only show the first 50 points
        # if i > 50:
        #    break
    
    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z = torch.cat(z, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    if use_var_decoder:
        x_rec_sigma = torch.cat(x_rec_sigma, dim=0).numpy()

    if use_var_decoder:
        # Grid for probability map
        n_points_axis = 50
        xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(
            z[:, 0].min(),
            z[:, 0].max(),
            z[:, 1].min(),
            z[:, 1].max(),
            n_points_axis
        )
        
        all_f_mu, all_f_sigma = [], []
        for z_grid in tqdm(z_grid_loader):
            
            z_grid = z_grid[0].to(device)

            with torch.inference_mode():
                mu_rec_grid = mu_decoder(z_grid)
                log_sigma_rec_grid = softclip(var_decoder(z_grid), min=-6)

            sigma_rec_grid = torch.exp(log_sigma_rec_grid)

            all_f_mu += [mu_rec_grid.cpu()]
            all_f_sigma += [sigma_rec_grid.cpu()]

        f_mu = torch.cat(all_f_mu, dim=0)
        f_sigma = torch.cat(all_f_sigma, dim=0)

        # get diagonal elements
        sigma_vector = f_sigma.mean(axis=1)

    # create figures
    if not os.path.isdir(f"figures/{path}"): os.makedirs(f"figures/{path}")

    plt.figure()
    if dataset == "mnist":
        for yi in np.unique(labels):
            idx = labels == yi
            plt.plot(z[idx, 0], z[idx, 1], 'x', ms=5.0, alpha=1.0)
    else:
        plt.plot(z[:, 0], z[:, 1], 'x', ms=5.0, alpha=1.0)

    if use_var_decoder:
        precision_grid = np.reshape(sigma_vector, (n_points_axis, n_points_axis))
        plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
        plt.colorbar()

    plt.savefig(f"figures/{path}/ae_contour.png")
    plt.close(); plt.cla()

    if dataset == "mnist":
        for i in range(min(len(z), 10)):
            nplots = 3 if use_var_decoder else 2

            plt.figure()
            plt.subplot(1,nplots,1)
            plt.imshow(x[i].reshape(28,28))

            plt.subplot(1,nplots,2)
            plt.imshow(x_rec_mu[i].reshape(28,28))

            if use_var_decoder:
                plt.subplot(1,nplots,3)
                plt.imshow(x_rec_sigma[i].reshape(28,28))

            plt.savefig(f"figures/{path}/ae_recon_{i}.png")
            plt.close(); plt.cla()


def train_ae(dataset = "mnist", use_var_decoder=False):

    # data
    train_loader, val_loader = get_data(dataset)

    # model
    model = LitAutoEncoder(dataset, use_var_decoder)

    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    # early stopping
    callbacks = [EarlyStopping(monitor="val_loss")]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_device, num_nodes=1, auto_scale_batch_size=True, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    
    # save weights
    path = f"{dataset}/ae_[use_var_dec={use_var_decoder}]"
    if not os.path.isdir(f"weights/{path}"): os.makedirs(f"weights/{path}")
    torch.save(model.encoder.state_dict(), f"weights/{path}/encoder.pth")
    torch.save(model.mu_decoder.state_dict(), f"weights/{path}/mu_decoder.pth")
    if use_var_decoder:
        torch.save(model.var_decoder.state_dict(), f"weights/{path}/var_decoder.pth")


if __name__ == "__main__":

    dataset = "mnist"
    train = True
    use_var_decoder = False

    # train or load auto encoder
    if train:
        train_ae(dataset, use_var_decoder=use_var_decoder)

    test_ae(dataset, use_var_decoder=use_var_decoder)

    