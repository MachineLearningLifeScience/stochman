# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:55:54 2018

@author: nsde
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math


class AE_bivariate(nn.Module):
    def __init__(self, latent_size, device="cpu", learning_rate=1e-4):
        super(AE_bivariate, self).__init__()
        self.latent_size = latent_size

        # self.encoder = nn.Sequential(nn.Linear(2, 512),
        #                                     nn.ReLU(),
        #                                     nn.Linear(512, 256),
        #                                     nn.ReLU(),
        #                                     nn.Linear(256, latent_size))
        #
        # self.decoder = nn.Sequential(nn.Linear(latent_size, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 256),
        #                             nn.ReLU(),
        #                             nn.Linear(256, 2))

        self.encoder = nn.Sequential(nn.Linear(2, 50),
                                    nn.Tanh(),
                                    nn.Linear(50, latent_size))

        self.decoder = nn.Sequential(nn.Linear(latent_size, 50),
                                    nn.Tanh(),
                                    nn.Linear(50, 2))

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)
        if device == "gpu":
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x, switch):
        x = x.view(x.shape[0], -1)
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

    def loss_function(self, x, x_rec):
        loss = self.criterion(x, x_rec)
        return loss

    def fit(self, X, n_epochs=100, learning_rate=1e-4, batch_size=64, verbose=True, labels=None):

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)
        n_batch = int(np.ceil(X.shape[0] / batch_size))

        for e in range(1, n_epochs + 1):
            loss_avg = 0
            x_enc = []
            for idx in range(n_batch):
                self.optimizer.zero_grad()
                x = X[idx * batch_size:(idx + 1) * batch_size]
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
                s = 1.0 if e > n_epochs / 2 else 0.0  # warmup the mean function for half the number of epochs
                x_rec, z = self.forward(x, s)
                loss = self.loss_function(x, x_rec)

                if torch.isnan(loss):
                    print(x, x_rec, z)
                    import sys
                    sys.exit()

                loss.backward()
                self.optimizer.step()
                loss_avg += loss.item()

                if verbose and self.latent_size == 2:
                    x_enc.append(z)

            if verbose:
                print("Epoch {0}/{1}, Loss {2}".format(e, n_epochs, loss_avg / X.shape[0]))
                # save_image(torch.cat((x[:10].view((10, *self.image_dim)), x_mu[:10].view(10, *self.image_dim)),
                #            dim=0).cpu(), '../images/recon_'+str(e) + '.png', nrow=10)
                #
                # if self.latent_size == 2:
                #     # Save grid of images
                #     z = torch.stack([array.flatten() for array in torch.meshgrid(
                #         [torch.linspace(-3, 3, 40), torch.linspace(-3, 3, 40)])]).t()
                #     x_mu = self.dec_mu(z.to(self.device))
                #     save_image(x_mu.view(1600, *self.image_dim).cpu(), '../images/grid_'+str(e)+'.png', nrow=40)
                #     # Save point cloud
                #     x_enc = torch.cat(x_enc, dim=0).detach().cpu().numpy()
                #     fig=plt.figure(frameon=False)
                #     plt.scatter(x_enc[:,0], x_enc[:,1], c=labels)
                #     if labels is not None: plt.colorbar()
                #     plt.savefig('../images/latent_'+str(e)+'.png')
                #     plt.close(fig)

    def intrinsic_coordinates(self, points):  # encoder
        points = points.to(self.device)
        z_mu = self.enc_mu(points)
        z_var = self.enc_var(points)
        coors = torch.cat((z_mu, z_var), dim=1)
        return coors.cpu()

    def embed(self, coords):  # decoder
        coords = coords.to(self.device)
        points_mu = self.dec_mu(coords)
        points_var = self.dec_var(coords)

        print("mu:", points_mu.shape)
        print("var:", points_var.shape)

        points = torch.cat((points_mu, points_var))

        print("points:", points.shape)
        return points.cpu()

    def curve_energy(self, coords):
        # coords: Nx(D-1)
        points = self.embed(coords)  # NxD
        energy = (points[1:] - points[:-1]).pow(2).sum()
        return energy
