"""
In this script we visualize geodesics in the latent space
of a Variational Autoencoder (VAE) trained on motion-capture data.
This VAE decodes to a product of von Mises-Fisher distributions,
and has been modified to extrapolate to uncertainty away from the
support of the data. For more information, see [1, Sec. 4.3.].

Being more precise, this script replicates Fig. 6 (Left) of [1].
We rely on the tooling provided in the implementation of the
hyperspherical VAE [2]. Our proposed pullback of the Fisher-Rao
relies on local computations of the KL divergence between the 
decoded distributions, and we compute this quantity for vMF distributions
using Monte Carlo integration (see ./vmf.py)

[1] Pulling back information geometry, by Georgios Arvanitidis,
    Miguel González-Duque, Alison Pouplin, Dimitris Kalatzis and
    Søren Hauberg.

[2] Hyperspherical Variational Auto-Encoders, by Tim R. Davidson,
    Luca Falorsi, Nicola De Cao, Thomas Kipf and Jakub M. Tomczak.
    https://github.com/nicola-decao/s-vae-pytorch.
"""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from stochman.manifold import StatisticalManifold
from stochman.discretized_manifold import DiscretizedManifold

from vae_motion import VAE_Motion
from vmf import VonMisesFisher
from data_utils import load_bones_data

MODELS_DIR = Path(__file__).parent.resolve() / "models"


class TranslatedSigmoid(nn.Module):
    """
    A translated sigmoid function that is used
    to regulate entropy in entropy networks.

    Input:
        - beta: float (1.5 by default). The lower,
          the narrower the region of low entropy.
          Negative values give nice latent spaces
    """

    def __init__(self, beta: float = -1.5) -> None:
        super(TranslatedSigmoid, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, x):
        beta = torch.nn.functional.softplus(self.beta)
        alpha = -beta * (6.9077542789816375)
        val = torch.sigmoid((x + alpha) / beta)

        return val


class VAE_motion_UQ(VAE_Motion):
    def __init__(self, training_data: torch.Tensor, n_bones: int, n_hidden: int, radii: torch.Tensor) -> None:
        super().__init__(2, n_bones, n_hidden, radii)

        self.training_data = training_data
        self.n_clusters = 500
        self.beta = -5.5
        self.limit_k = 0.1

        self.translated_sigmoid = None
        self.encodings = None
        self.cluster_centers = None

    def update_cluster_centers(self):
        """
        Encodes the training data and
        runs KMeans to get centers.
        """
        # Defining the translated sigmoid used to calibrate
        # the uncertainty (see Eq. 26 of [1]).
        self.translated_sigmoid = TranslatedSigmoid(self.beta)

        # Fitting K Means on the training encodings.
        batch_size, n_bones, _ = self.training_data.shape
        flat_training_data = self.training_data.reshape(batch_size, n_bones * 3)
        self.encodings, _ = self.encode(flat_training_data)
        Z = self.encodings.detach().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(Z)
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

    def min_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        D(z) in the notation of [1, Sec. C.2.]
        """
        zsh = z.shape
        z = z.view(-1, z.shape[-1])  # Nx(zdim)

        z_norm = (z ** 2).sum(1, keepdim=True)  # Nx1
        center_norm = (self.cluster_centers ** 2).sum(1).view(1, -1)  # 1x(num_clusters)
        d2 = (
            z_norm + center_norm - 2.0 * torch.mm(z, self.cluster_centers.transpose(0, 1))
        )  # Nx(num_clusters)
        d2.clamp_(min=0.0)  # Nx(num_clusters)
        min_dist, _ = d2.min(dim=1)  # N

        return min_dist.view(zsh[:-1])

    def reweight(self, z: torch.Tensor) -> torch.Tensor:
        """
        An alternate version of the decoder that pushes
        k to self.limit_k away of the support of the
        data. See Eq. 27 of [1, Sec. C.2.].
        """
        zsh = z.shape
        z = z.reshape(-1, zsh[-1])
        dec_mu, dec_k = super().decode(z)  # Nx(num_bones)x3, Nx(num_bones)

        # Distance to the supp.
        alpha = self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)

        reweighted_k = (1 - alpha) * dec_k + alpha * (torch.ones_like(dec_k) * self.limit_k)

        mush = dec_mu.shape
        ksh = dec_k.shape
        return dec_mu.view(zsh[:-1] + mush[1:]), reweighted_k.view(zsh[:-1] + ksh[1:])

    def decode(self, z, reweight=True) -> VonMisesFisher:
        if reweight:
            mu, k = self.reweight(z)
        else:
            mu, k = super().decode(z)

        return VonMisesFisher(loc=mu, scale=k)

    def forward(self, x):
        """
        Forward pass through the network, w. extrapolation
        of k to self.limit_k.

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
        p_mu, p_k = self.reweight(z)

        return x, p_mu, p_k.unsqueeze(2), q_mu, q_var

    def plot_latent_space(self, ax=None):
        """
        Visualizes the latent space, illuminating it with
        the average concentration parameter of the decoded
        vMFs.
        """
        encodings = self.encodings.detach().numpy()
        enc_x, enc_y = encodings[:, 0], encodings[:, 1]

        n_x, n_y = 300, 300
        x_lims = (enc_x.min() - 0.1, enc_x.max() + 0.1)
        y_lims = (enc_y.min() - 0.1, enc_y.max() + 0.1)
        z1 = torch.linspace(*x_lims, n_x)
        z2 = torch.linspace(*y_lims, n_x)

        K = np.zeros((n_y, n_x))
        zs = torch.Tensor([[x, y] for x in z1 for y in z2])
        positions = {
            (x.item(), y.item()): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }
        decoded_dist = self.decode(zs)
        ks = decoded_dist.scale
        ks = ks.detach().numpy()
        mean_ks = np.mean(ks, axis=1)
        for l, (x, y) in enumerate(zs):
            i, j = positions[(x.item(), y.item())]
            K[i, j] = mean_ks[l]

        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.scatter(encodings[:, 0], encodings[:, 1], s=8, alpha=0.75, c="k", edgecolors="white")
        plot = ax.imshow(K, extent=[*x_lims, *y_lims], cmap="Blues_r")
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)


if __name__ == "__main__":
    # Loading the dataset.
    bones = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    train_dataset, test_dataset, radii = load_bones_data(bones=bones)
    n_bones = len(bones)

    # Loading the model.
    n_hidden = 30
    vae_uq = VAE_motion_UQ(train_dataset.tensors[0], n_bones, n_hidden, radii)
    vae_uq.load_state_dict(torch.load(MODELS_DIR / "motion_2.pt"))

    # Calibrating its uncertainty. In this example, we extrapolate to uncertain
    # vMFs (i.e. the concentration parameter kappa -> 0)
    vae_uq.update_cluster_centers()

    # Adding curve energy and length using our approximations of the
    # Fisher-Rao pullback metric.
    vae_manifold = StatisticalManifold(vae_uq)

    # Defining a discrete approximation using a graph whose edges
    # are given by the curve energy between them.
    grid = [torch.linspace(-2.5, 2.5, 50), torch.linspace(-2.5, 2.5, 50)]
    discrete_manifold_approximation = DiscretizedManifold()
    discrete_manifold_approximation.fit(vae_manifold, grid)

    # Visualizing the latent space and several geodesics.
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    vae_uq.plot_latent_space(ax=ax)
    for _ in range(30):
        idx_1, idx_2 = np.random.randint(0, len(vae_uq.encodings), size=(2,))
        geodesic, _ = discrete_manifold_approximation.connecting_geodesic(
            vae_uq.encodings[idx_1], vae_uq.encodings[idx_2]
        )
        geodesic.plot(ax=ax, c="green", linewidth=2)
    ax.axis("off")

    fig.savefig(
        Path(__file__).parent.resolve() / "vmf_latent_space_with_geodesics.jpg", dpi=120, bbox_inches="tight"
    )

    plt.show()
