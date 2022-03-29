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


class VAE_Motion(torch.nn.Module):
    def __init__(self, n_bones: int, n_hidden: int, radii: torch.Tensor) -> None:
        super().__init__()
        self.n_bones = n_bones
        self.n_hidden = n_hidden
        self.input_dim = n_bones * 3
        self.radii = torch.from_numpy(radii).unsqueeze(0).type(torch.float)

        # An encoder for N(mu, sigma) in latent space (dim 2)
        self.encoder = nn.Linear(self.input_dim, self.n_hidden)
        self.enc_mu = nn.Linear(self.n_hidden, 2)
        self.enc_logsigma = nn.Linear(self.n_hidden, 2)

        # A decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, self.n_hidden),
            nn.Linear(self.n_hidden, self.input_dim),
        )
        self.dec_mu = nn.Linear(self.input_dim, self.input_dim)
        self.dec_k = nn.Sequential(nn.Linear(self.input_dim, self.n_bones), nn.Softplus())

        # A prior over the latent codes
        self.p_z = Normal(torch.zeros(2), torch.ones(2))

    def encode(self, x: torch.Tensor) -> Normal:
        hidden = self.encoder(x)
        mu, logsigma = self.enc_mu(hidden), self.enc_logsigma(hidden)

        return Normal(loc=mu, scale=torch.exp(logsigma))

    def decode(self, z: torch.Tensor) -> VonMisesFisher:
        hidden = self.decoder(z)
        mu, k = self.dec_mu(hidden), self.dec_k(hidden)

        # Shape back into a circle and normalize
        batch_size, inp = mu.shape
        mu = mu.view(batch_size, inp // 3, 3)
        norm_mu = mu.norm(dim=-1, keepdim=True)
        mu = torch.div(mu, norm_mu)

        # Avoid collapses
        k = k + 0.01

        return VonMisesFisher(loc=mu, scale=k.unsqueeze(-1))

    def forward(self, x: torch.Tensor):
        """
        Assuming x is a collection of points in the sphere
        like (b, n, 3)
        """
        # Flattening x
        b, n, _ = x.shape
        assert n == self.n_bones
        x_flat = x.reshape(b, n * 3)

        # Encoding
        q_z_given_x = self.encode(x_flat)
        z = q_z_given_x.sample()

        # Decoding
        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

    def elbo_loss(self, x: torch.Tensor, q_z_given_x: Normal, p_x_given_z: VonMisesFisher):
        rec_loss = -(self.radii * p_x_given_z.log_prob(x)).sum(dim=1)
        kl = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)

        beta = 0.01
        return (rec_loss + beta * kl).sum()
