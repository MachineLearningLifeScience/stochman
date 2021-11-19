#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ae_models import AE_mnist

from laplace import Laplace

n_epochs = 50
batch_size = 128  # full batch
true_sigma_noise = 0.3

# create mnist data set
load = False
filename = '../models/vae_normal'

mnist = MNIST('../data/', download=True)
X_train = mnist.train_data.reshape(-1, 784).numpy() / 255.0
y_train = torch.from_numpy(mnist.train_labels.numpy())

X_test = mnist.test_data.reshape(-1, 784).numpy() / 255.0
y_test = torch.from_numpy(mnist.test_labels.numpy())

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AE_mnist(latent_size=2, device="gpu" if torch.cuda.is_available() else "cpu")
model.fit(X_train, n_epochs=n_epochs, learning_rate=1e-3, batch_size=batch_size, verbose=True, labels=y_train)

# Visualize Latent Space
X_test_fold = X_test.view(X_test.shape[0], -1).to(device)
z_test = model.encoder(X_test_fold)
z_test = z_test.detach()

# Laplace Approximation
la = Laplace(model.decoder, 'regression', subset_of_weights='last_layer', hessian_structure='diag')

# Getting Z representations for X_train
X_fold = X_train.view(X_train.shape[0], -1).to(device)
model.eval()
with torch.inference_mode():
    z = model.encoder(X_fold)
    x_rec = model.decoder(z)
z_loader = DataLoader(TensorDataset(z, x_rec), batch_size=batch_size)
# Fitting
la.fit(z_loader)
log_prior, log_sigma = torch.ones(1, requires_grad=True, device="cuda:0"), torch.ones(1, requires_grad=True, device="cuda:0")
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-2)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

X_test_fold = X_test.view(X_test.shape[0], -1).to(device)
z_test = model.encoder(X_test_fold)
z_test = z_test.detach()

# GRID FOR PROBABILITY MAP
n_points_axis = 50
zx_grid = np.linspace(z_test[:,0].min().cpu().detach().numpy() - 1.5, z_test[:,0].max().cpu().detach().numpy() + 1.5, n_points_axis)
zy_grid = np.linspace(z_test[:,1].min().cpu().detach().numpy() - 1.5, z_test[:,1].max().cpu().detach().numpy() + 1.5, n_points_axis)

xg_mesh, yg_mesh = np.meshgrid(zx_grid, zy_grid)
xg = xg_mesh.reshape(n_points_axis ** 2, 1)
yg = yg_mesh.reshape(n_points_axis ** 2, 1)
Z_grid_test = np.hstack((xg, yg))
Z_grid_test = torch.from_numpy(Z_grid_test).float().detach().to(device)

all_f_mu, all_f_sigma = [], []
for i in tqdm(range(Z_grid_test.shape[0])):
    f_mu, f_var = la(Z_grid_test[i:i+1,:])
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()

    all_f_mu.append(f_mu)
    all_f_sigma.append(f_sigma)

f_mu = np.stack(all_f_mu)
f_sigma = np.stack(all_f_sigma)

sigma_vector = f_sigma[:,np.arange(f_sigma.shape[1]), np.arange(f_sigma.shape[1])].mean(axis=1)

plt.figure()
plt.plot(z_test[:, 0].cpu(), z_test[:, 1].cpu(), 'wx', ms=5.0, alpha=1.0)
precision_grid = np.reshape(sigma_vector, (n_points_axis,n_points_axis))
plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
plt.colorbar()
plt.savefig("mnist.png")
plt.show()
 
