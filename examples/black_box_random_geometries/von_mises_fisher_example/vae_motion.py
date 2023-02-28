"""
This script contains a VAE that decodes to a vMF
distribution to model motion capture data.
It will be trained in train.py, and analyzed in main.py
for defining geodesics in latent space
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from vmf import VonMisesFisher

from stochman.manifold import StatisticalManifold
from stochman.nnj import Norm2


class VAE_motion(torch.nn.Module):
    """
    A VAE that decodes to a product of vMF distriutions
    """

    def __init__(self, latent_dim, n_bones=30, n_hidden=30, radii=None):
        super(VAE_motion, self).__init__()

        self.latent_dim = latent_dim
        self.n_bones = n_bones
        self.n_hidden = n_hidden
        if radii is None:
            self.radii = torch.ones(1, n_bones)
        else:
            self.radii = torch.from_numpy(radii).view(1, -1)

        input_dim = n_bones * 3
        enc_layers = [
            nn.Linear(input_dim, self.n_hidden),
        ]  # activation()]
        # for inputs, outputs in zip(n_hidden[:-1], n_hidden[1:]):
        #     enc_layers.extend([nn.Linear(inputs, outputs), activation()])

        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(self.n_hidden, latent_dim)
        self.var = nn.Sequential(nn.Linear(self.n_hidden, latent_dim), nn.Softplus())

        # dec_defs = n_hidden.copy() + [latent_dim]
        # dec_defs.reverse()
        dec_layers = [
            *[
                nn.Linear(latent_dim, n_hidden),
            ],  # activation()],
            *[nn.Linear(n_hidden, input_dim)],
        ]
        # for inputs, outputs in zip(dec_defs[:-1], dec_defs[1:]):
        #     dec_layers.extend([nn.Linear(inputs, outputs), activation()])

        self.decoder = nn.Sequential(*dec_layers)
        # Von Mises - Fisher mean and concentration
        self.dec_mu = nn.Sequential(nn.Linear(input_dim, input_dim))
        self.dec_k = nn.Sequential(nn.Linear(input_dim, self.n_bones), nn.Softplus())

        self.norm_2 = Norm2(dim=2)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        var = self.var(h)

        return mu, var

    def reparameterize_enc(self, mu, var):
        return mu + var.sqrt() * torch.randn_like(var)

    def decode(self, z) -> VonMisesFisher:
        """
        Returns the parameters mu and k of
        the decoded vMF distribution.
        """
        # A hidden layer of the decoder
        h = self.decoder(z)

        # A layer that learns the mean of the vMF
        mu = self.dec_mu(h)

        # Reshaping it to b x n_bones x 3, and normalizing
        # it so that it lives in the sphere
        batch_size, inp = mu.shape
        mu = mu.view(batch_size, inp // 3, 3)
        norm_mu = self.norm_2(mu).sqrt()
        mu = torch.div(mu, norm_mu)

        # A layer that learns the concentration of the vMF
        k = self.dec_k(h) + 0.01  # avoid collapses and singularities

        # Building a product of n_bones independent
        # distributions.
        vMF = VonMisesFisher(loc=mu, scale=k)
        return vMF

    def forward(self, x):
        """
        Forward pass through the network.

        Returns (x, q_mu, q_var, p_mu, p_k) where
        q is the Normal in latent space and p is the vMF
        in data space.
        """
        # Flattens the bones.
        batch_size, n_bones, _ = x.shape
        x_flat = x.reshape(batch_size, n_bones * 3)

        # Gets the parameters of the Gaussian in latent space.
        q_mu, q_var = self.encode(x_flat)

        # Samples from those Gaussians
        z = self.reparameterize_enc(q_mu, q_var)

        # Decodes the samples
        vmf_dist = self.decode(z)
        p_mu = vmf_dist.loc
        p_k = vmf_dist.scale

        return x, p_mu, p_k.unsqueeze(2), q_mu, q_var

    # This could be a static method.
    def elbo_loss(self, x, p_mu, p_k, q_mu, q_var):
        log_pxz = self.radii * VonMisesFisher(p_mu, p_k).log_prob(x)
        KLD = -0.5 * torch.sum(1 + q_var.log() - q_mu.pow(2) - q_var)

        beta = 0.01
        return torch.sum(-log_pxz + beta * KLD)
