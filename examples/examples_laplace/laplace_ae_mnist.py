#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ae_models import AE_mnist

from laplace import Laplace

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath} \usepackage{marvosym}')
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})

N = 50000
N_test = 300
n_epochs = 30
batch_size = 128  # full batch
true_sigma_noise = 0.3

# create mnist data set
load = False
filename = '../models/vae_normal'

mnist = MNIST('../data/', download=True)
X_train = mnist.train_data.reshape(-1, 784).numpy() / 255.0
y_train = mnist.train_labels.numpy()

X_test = mnist.test_data.reshape(-1, 784).numpy() / 255.0
y_test = mnist.test_labels.numpy()

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size)

model = AE_mnist(latent_size=2)
model.fit(X_train, n_epochs=n_epochs, learning_rate=1e-3, batch_size=batch_size, verbose=True, labels=y_train)

# Visualize Latent Space
X_test_fold = X_test.view(X_test.shape[0], -1)
z_test = model.encoder(X_test_fold)
z_test = z_test.detach()

# Laplace Approximation
la = Laplace(model.decoder, 'regression', subset_of_weights='all', hessian_structure='full')

# Getting Z representations for X_train
X_fold = X_train.view(X_train.shape[0], -1)
model.eval()
with torch.inference_mode():
    z = model.encoder(X_fold)
    x_rec = model.decoder(z)
z_loader = DataLoader(TensorDataset(z, x_rec), batch_size=batch_size)
# Fitting
la.fit(z_loader)
log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-2)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

plt.figure()
fig, ax = plt.subplots(dpi=150)

X_test_fold = X_test.view(X_test.shape[0], -1)
z_test = model.encoder(X_test_fold)
z_test = z_test.detach()

# GRID FOR PROBABILITY MAP
n_points_axis = 50
zx_grid = np.linspace(z_test[:,0].min().detach().numpy() - 1.5, z_test[:,0].max().detach().numpy() + 1.5, n_points_axis)
zy_grid = np.linspace(z_test[:,1].min().detach().numpy() - 1.5, z_test[:,1].max().detach().numpy() + 1.5, n_points_axis)

xg_mesh, yg_mesh = np.meshgrid(zx_grid, zy_grid)
xg = xg_mesh.reshape(n_points_axis ** 2, 1)
yg = yg_mesh.reshape(n_points_axis ** 2, 1)
Z_grid_test = np.hstack((xg, yg))
Z_grid_test = torch.from_numpy(Z_grid_test).float().detach()

f_mu, f_var = la(Z_grid_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()

sigma_vector = f_sigma[:,0,0]

plt.plot(z_test[:, 0], z_test[:, 1], 'wx', ms=5.0, alpha=1.0)

precision_grid = np.reshape(sigma_vector, (n_points_axis,n_points_axis))
plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
plt.colorbar()

plt.title(f'Data N={N}')
plt.show()

