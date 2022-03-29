import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from stochman.manifold import StatisticalManifold

from vae_motion import VAE_Motion
from vmf import VonMisesFisher
from data_utils import load_bones_data


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


class VAE_motion_UQ(VAE_Motion, StatisticalManifold):
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
        self.translated_sigmoid = TranslatedSigmoid(self.beta)

        batch_size, n_bones, _ = self.training_data.shape
        flat_training_data = self.training_data.reshape(batch_size, n_bones * 3)
        self.encodings, _ = self.encode(flat_training_data)
        Z = self.encodings.detach().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(Z)
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

    def min_distance(self, z: torch.Tensor) -> torch.Tensor:
        """V(z) in the notation of the paper"""
        # What's the size of z?
        # |z| = (batch, zdim), right?
        zsh = z.shape
        z = z.view(-1, z.shape[-1])  # Nx(zdim)

        z_norm = (z ** 2).sum(1, keepdim=True)  # Nx1
        center_norm = (self.cluster_centers ** 2).sum(1).view(1, -1)  # 1x(num_clusters)
        d2 = (
            z_norm + center_norm - 2.0 * torch.mm(z, self.cluster_centers.transpose(0, 1))
        )  # Nx(num_clusters)
        d2.clamp_(min=0.0)  # Nx(num_clusters)
        min_dist, _ = d2.min(dim=1)  # N
        # from IPython import embed; embed()
        return min_dist.view(zsh[:-1])

    def similarity(self, v: torch.Tensor) -> torch.Tensor:
        """
        T(z) or alpha in the notation of the paper, but
        backwards and translated to make it 0 at 0 and
        1 and infty.
        """
        return self.translated_sigmoid(v)

    def reweight(self, z: torch.Tensor) -> torch.Tensor:
        """
        An alternate version of the decoder that pushes
        k to self.limit_k away of the support of the
        data.
        """
        zsh = z.shape
        z = z.reshape(-1, zsh[-1])
        dec_mu, dec_k = super().decode(z)  # Nx(num_bones)x3, Nx(num_bones)

        # Distance to the supp.
        alpha = self.similarity(self.min_distance(z)).unsqueeze(-1)

        reweighted_k = (1 - alpha) * dec_k + alpha * (torch.ones_like(dec_k) * self.limit_k)

        mush = dec_mu.shape
        ksh = dec_k.shape
        return dec_mu.view(zsh[:-1] + mush[1:]), reweighted_k.view(zsh[:-1] + ksh[1:])

    def decode(self, z, reweight=True):
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
        _, ks = self.reweight(zs)  # [b, 28, 28]
        ks = ks.detach().numpy()
        mean_ks = np.mean(ks, axis=1)
        for l, (x, y) in enumerate(zs):
            i, j = positions[(x.item(), y.item())]
            K[i, j] = mean_ks[l]

        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.scatter(encodings[:, 0], encodings[:, 1], s=1)
        # ax.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1])
        plot = ax.imshow(K, extent=[*x_lims, *y_lims])
        plt.colorbar(plot, ax=ax)
        # plt.show()


if __name__ == "__main__":
    # Loading the model.

    n_hidden = 30
    bones = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    train_dataset, test_dataset, radii = load_bones_data(bones=bones)
    n_bones = len(bones)

    vae_uq = VAE_motion_UQ(train_dataset.tensors[0], n_bones, n_hidden, radii)
    vae_uq.load_state_dict(
        torch.load("./examples/black_box_random_geometries/von_mises_fisher_example/models/motion_2.pt")
    )
    vae_uq.update_cluster_centers()
    vae_manifold = StatisticalManifold(vae_uq)

    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    vae_uq.plot_latent_space(ax=ax)
    for _ in range(10):
        idx_1, idx_2 = np.random.randint(0, len(vae_uq.encodings), size=(2,))
        geodesic, _ = vae_manifold.connecting_geodesic(vae_uq.encodings[idx_1], vae_uq.encodings[idx_2])
        geodesic.plot(ax=ax)

    plt.show()
