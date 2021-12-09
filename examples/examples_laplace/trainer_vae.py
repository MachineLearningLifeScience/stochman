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

class LitVariationalAutoEncoder(pl.LightningModule):
    def __init__(self, dataset, use_var_decoder):
        super().__init__()

        # scaling of kl term
        self.alpha = 1e-4
        self.use_var_decoder = use_var_decoder

        latent_size = 2
        self.mu_encoder = get_encoder(dataset, latent_size)
        self.var_encoder = get_encoder(dataset, latent_size)

        self.mu_decoder = get_decoder(dataset, latent_size)

        if self.use_var_decoder:
            self.var_decoder = get_decoder(dataset, latent_size)

    def forward(self, x):
        mean = self.mu_encoder(x)
        log_sigma = softclip(self.var_encoder(x), min=-3)
        return mean, log_sigma

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z_mu, z_log_sigma = self.forward(x)

        z_sigma = torch.exp(z_log_sigma)
        z = z_mu + torch.randn_like(z_sigma) * z_sigma

        mu_x_hat = self.mu_decoder(z)

        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3)
        
            # reconstruction term: 
            rec = (torch.pow((mu_x_hat - x) / torch.exp(log_sigma_x_hat), 2) + log_sigma_x_hat).mean()
        else:
            rec = F.mse_loss(mu_x_hat, x)

        # kl term
        kl = -0.5 * torch.sum(1 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)
        
        self.log('train_loss', rec + self.alpha * kl)
        self.log('reconstruciton_loss', rec)
        self.log('kl_loss', kl)

        return rec + self.alpha * kl

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        
        z_mu, z_log_var = self.forward(x)

        z_sigma = torch.exp(z_log_var).sqrt()
        z = z_mu + torch.randn_like(z_sigma) * z_sigma

        mu_x_hat = self.mu_decoder(z)
        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3)

            # reconstruction term: 
            rec = (torch.pow((mu_x_hat - x) / torch.exp(log_sigma_x_hat), 2) + log_sigma_x_hat).mean()

        else:
            # reconstruction term
            rec = F.mse_loss(mu_x_hat, x)

        # kl term
        kl = -0.5 * torch.sum(1 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)

        self.log('val_loss', rec + self.alpha * kl)
        self.log('val_reconstruciton_loss', rec)
        self.log('val_kl_loss', kl)


def test_vae(dataset, batch_size=1, use_var_decoder=False):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{dataset}/vae_[use_var_dec={use_var_decoder}]"
    
    latent_size = 2
    mu_encoder = get_encoder(dataset, latent_size).eval().to(device)
    var_encoder = get_encoder(dataset, latent_size).eval().to(device)
    mu_encoder.load_state_dict(torch.load(f"weights/{path}/mu_encoder.pth"))
    var_encoder.load_state_dict(torch.load(f"weights/{path}/var_encoder.pth"))

    mu_decoder = get_decoder(dataset, latent_size).eval().to(device)
    mu_decoder.load_state_dict(torch.load(f"weights/{path}/mu_decoder.pth"))

    if use_var_decoder:
        var_decoder = get_decoder(dataset, latent_size).eval().to(device)
        var_decoder.load_state_dict(torch.load(f"weights/{path}/var_decoder.pth"))

    train_loader, val_loader = get_data(dataset, batch_size)

    # forward eval
    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = [], [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.view(xi.size(0), -1).to(device)
        with torch.inference_mode():

            z_mu_i = mu_encoder(xi)
            z_log_sigma_i = softclip(var_encoder(xi), min=-3)
            z_sigma_i = torch.exp(z_log_sigma_i)

            if use_var_decoder:
                
                # sample from distribution
                zi = z_mu_i + torch.randn_like(z_sigma_i) * z_sigma_i

                mu_rec_i = mu_decoder(zi)
                log_sigma_rec_i = softclip(var_decoder(zi), min=-3)
                sigma_rec_i = torch.exp(log_sigma_rec_i)

            else:
                
                # if we only have one decoder, then we can obtain uncertainty 
                # estimates by mc sampling from latent space

                # number of mc samples
                N = 30

                mu_rec_i, mu2_rec_i = None, None
                for n in range(N):

                    # sample from distribution
                    zi = z_mu_i + torch.randn_like(z_sigma_i) * z_sigma_i
                    x_reci = mu_decoder(zi)

                    # compute running mean and running variance
                    if mu_rec_i is None:
                        mu_rec_i = x_reci
                        mu2_rec_i = x_reci**2
                    else:
                        mu_rec_i += x_reci
                        mu2_rec_i += x_reci**2

                mu_rec_i = mu_rec_i / N
                mu2_rec_i = mu2_rec_i / N
                sigma_rec_i = (mu2_rec_i - mu_rec_i**2)**0.5

            x += [xi.cpu()]
            z_mu += [z_mu_i.detach().cpu()]
            z_sigma += [z_sigma_i.detach().cpu()]
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

    if use_var_decoder:
        # Grid for probability map
        n_points_axis = 50
        xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(
            z_mu[:, 0].min(),
            z_mu[:, 0].max(),
            z_mu[:, 1].min(),
            z_mu[:, 1].max(),
            n_points_axis
        )
        
        all_f_mu, all_f_sigma = [], []
        for z_grid in tqdm(z_grid_loader):
            
            z_grid = z_grid[0].to(device)

            with torch.inference_mode():
                mu_rec_grid = mu_decoder(z_grid)
                log_sigma_rec_grid = softclip(var_decoder(z_grid), min=-3)

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
            plt.plot(z_mu[idx, 0], z_mu[idx, 1], 'x', ms=5.0, alpha=1.0)
    else:
        plt.plot(z_mu[:, 0], z_mu[:, 1], 'x', ms=5.0, alpha=1.0)

    if use_var_decoder:
        precision_grid = np.reshape(sigma_vector, (n_points_axis, n_points_axis))
        plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
        plt.colorbar()

    plt.savefig(f"figures/{path}/vae_contour.png")
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

            plt.savefig(f"figures/{path}/vae_recon_{i}.png")
            plt.close(); plt.cla()


def train_vae(dataset = "mnist", use_var_decoder=False):

    # data
    train_loader, val_loader = get_data(dataset)

    # model
    model = LitVariationalAutoEncoder(dataset, use_var_decoder)

    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    # early stopping
    callbacks = [EarlyStopping(monitor="val_loss")]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_device, num_nodes=1, auto_scale_batch_size=True, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    
    # save weights
    path = f"{dataset}/vae_[use_var_dec={use_var_decoder}]"
    if not os.path.isdir(f"weights/{path}"): os.makedirs(f"weights/{path}")
    torch.save(model.mu_encoder.state_dict(), f"weights/{path}/mu_encoder.pth")
    torch.save(model.var_encoder.state_dict(), f"weights/{path}/var_encoder.pth")
    torch.save(model.mu_decoder.state_dict(), f"weights/{path}/mu_decoder.pth")
    if use_var_decoder:
        torch.save(model.var_decoder.state_dict(), f"weights/{path}/var_decoder.pth")

if __name__ == "__main__":

    dataset = "mnist"
    train = True
    use_var_decoder = False

    # train or load auto encoder
    if train:
        train_vae(dataset, use_var_decoder=use_var_decoder)

    test_vae(dataset, use_var_decoder=use_var_decoder)

    