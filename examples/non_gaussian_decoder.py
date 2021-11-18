"""
This example uses the StochasticManifold class
to define geodesics in the latent space of a
non-Gaussian decoder.

TODO:
    - we'll need to minimize curve energy, do we already
      have a nice interface for doing that?
    - what about the things we'd like to install for the examples?
      (like, in this case, sklearn).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Poisson
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from stochman.manifold import StochasticManifold


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


class DecoderWithUQ(nn.Module):
    """
    An example of a decoder that returns a Poisson.
    You could have your own model (e.g. a full VAE),
    but the important part is implementing a `decode`
    method that returns a distribution.

    This decoder is purpusefully uncertain outside of
    a donut in R2. This uncertainty is represented by having
    an infinite variance lambda.
    """

    def __init__(self) -> None:
        super().__init__()

        self.mean = nn.Sequential(nn.Linear(2, 10), nn.Softplus())

        angles = torch.rand((100,)) * 2 * np.pi
        self.encodings = 3.0 * torch.vstack((torch.cos(angles), torch.sin(angles))).T
        kmeans = KMeans(n_clusters=70)
        kmeans.fit(self.encodings.detach().numpy())
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).type(torch.float)

        self.translated_sigmoid = TranslatedSigmoid(beta=-2.5)

    def decode(self, z: torch.Tensor) -> Poisson:
        """
        Uncertainty-aware decoder.
        """
        closeness = self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)
        dec_mean = self.mean(z)
        uncertain_mean = 1e5 * torch.ones_like(dec_mean)

        mean = (1 - closeness) * dec_mean + closeness * uncertain_mean
        return Poisson(rate=mean)

    def min_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        A function that measures the main distance w.r.t
        the cluster centers.

        TODO: change this to incorporate stochman.utilities.distance's.
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

    def plot_latent_space(self, ax=None, plot_points=True):
        """
        FOR DIMITRIS:

        To plot the mean entropy, you'll need to
        change the output of self.reweight to
        the distribution's parameters. I commented
        the relevant piece of code.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 7))

        encodings = self.encodings.detach().numpy()
        enc_x, enc_y = encodings[:, 0], encodings[:, 1]

        n_x, n_y = 300, 300
        x_lims = (enc_x.min() - 0.1, enc_x.max() + 0.1)
        y_lims = (enc_y.min() - 0.1, enc_y.max() + 0.1)
        z1 = torch.linspace(*x_lims, n_x)
        z2 = torch.linspace(*y_lims, n_x)

        entropy_K = np.zeros((n_y, n_x))
        zs = torch.Tensor([[x, y] for x in z1 for y in z2])
        positions = {
            (x.item(), y.item()): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        dist_ = self.decode(zs)
        entropy_ = dist_.rate.mean(dim=1)
        if len(entropy_.shape) > 1:
            # In some distributions, we decode
            # to a higher dimensional space.
            entropy_ = torch.mean(entropy_, dim=1)

        for l, (x, y) in enumerate(zs):
            i, j = positions[(x.item(), y.item())]
            entropy_K[i, j] = entropy_[l]

        if plot_points:
            ax.scatter(
                self.encodings[:, 0],
                self.encodings[:, 1],
                marker="o",
                c="w",
                edgecolors="k",
            )
        ax.imshow(entropy_K, extent=[*x_lims, *y_lims], cmap="Blues")


if __name__ == "__main__":
    dec = DecoderWithUQ()
    dec_manifold = StochasticManifold(dec)
    print(dec_manifold)
    zs = torch.randn(64, 2)
    rates = dec.decode(zs)

    dec.plot_latent_space()
    geodesic, success = dec_manifold.connecting_geodesic(dec.encodings[0], dec.encodings[1])
    print(success)
    geodesic.plot()
    plt.show()
