import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from examples.examples_laplace.vae_models import VAE_bivariate
from examples.examples_laplace.ae_models import AE_bivariate

from laplace import Laplace

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath} \usepackage{marvosym}')
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})

N = 10000
N_test = 300
n_epochs = 40
batch_size = 128  # full batch
true_sigma_noise = 0.3

# create simple sinusoid data set
def swiss_roll_2d(noise=0.5, n_samples=100):
  z = 2.0 * np.pi * (1 + 2 * np.random.rand(n_samples))
  x = z * np.cos(z) + noise*np.random.randn(n_samples)
  y = z * np.sin(z) + noise*np.random.randn(n_samples)
  return np.stack([x,y]).T, z

X_train, y_train = swiss_roll_2d(n_samples=N)
X_test, y_test = swiss_roll_2d(n_samples=N_test)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size)

model = AE_bivariate(latent_size=2)
model.fit(X_train, n_epochs=n_epochs, learning_rate=1e-3, batch_size=batch_size, verbose=True, labels=y_train)

# Laplace Approximation
la = Laplace(model.decoder, 'regression', subset_of_weights='all', hessian_structure='diag')

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

print(log_prior)
print(log_sigma)
#x_rec, z = model(X_test, 1.0)
from matplotlib.patches import Ellipse

#plot_type = 'Circles' # or 'Heatmap'
plot_type = 'Heatmap'

plt.figure()
fig, ax = plt.subplots(dpi=150)

if plot_type == 'Circles':

    # LA Prediction
    X_test_fold = X_test.view(X_test.shape[0], -1)
    z_test = model.encoder(X_test_fold)
    z_test = z_test.detach()

    f_mu, f_var = la(z_test)
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()

    plt.plot(X_train[:, 0], X_train[:, 1], 'bx', alpha=0.1)
    # plt.plot(X_test[:,0], X_test[:,1], '+')
    # plt.plot(x_rec.detach().numpy()[:,0], x_rec.detach().numpy()[:,1], 'gx', ms=3.0)
    for mu, sigma in zip(f_mu, f_sigma):
        r = 3*sigma[0, 0] # 3 x standard dev. is around 99.9% of uncertainty, right?
        circle = plt.Circle(xy=(mu[0], mu[1]), radius=r, fill=False, color='r', zorder=0)
        ax.add_patch(circle)

elif plot_type == 'Heatmap':
    # GRID FOR PROBABILITY MAP
    n_points_axis = 50
    x_grid = np.linspace(X_train[:,0].min().detach().numpy() - 5.0, X_train[:,0].max().detach().numpy() + 5.0, n_points_axis)
    y_grid = np.linspace(X_train[:,1].min().detach().numpy() - 5.0, X_train[:,1].max().detach().numpy() + 5.0, n_points_axis)

    xg_mesh, yg_mesh = np.meshgrid(x_grid, y_grid)
    xg = xg_mesh.reshape(n_points_axis ** 2, 1)
    yg = yg_mesh.reshape(n_points_axis ** 2, 1)
    X_grid_test = np.hstack((xg, yg))
    X_grid_test = torch.from_numpy(X_grid_test).float()

    # LA Prediction
    X_test_fold = X_grid_test.view(X_grid_test.shape[0], -1)
    z_test = model.encoder(X_test_fold)
    z_test = z_test.detach()

    f_mu, f_var = la(z_test)
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()

    #print(xg_mesh)
    #print(f_mu)

    plt.plot(X_train[:, 0], X_train[:, 1], 'bx', alpha=0.1)
    for mu, sigma in zip(f_mu, f_sigma):
        r = sigma[0, 0]
        circle = plt.Circle(xy=(mu[0], mu[1]), radius=r, fill=False, color='r')
        ax.add_patch(circle)
    #plt.show()

    # sigma_vector = f_sigma[:,0,0]
    #
    # plt.plot(X_train[:,0], X_train[:,1], 'rx', alpha=0.3)
    #
    # precision_grid = np.reshape(sigma_vector, (n_points_axis,n_points_axis))
    # plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
    # plt.colorbar()


plt.title(f'Data N={N}')
plt.show()

