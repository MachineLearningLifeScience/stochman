import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from laplace import Laplace
from data import get_data
from ae_models import get_encoder, get_decoder

import dill


def save_laplace(la, filepath):
    with open(filepath, 'wb') as outpt:
        dill.dump(la, outpt)


def load_laplace(filepath):
    with open(filepath, 'rb') as inpt:
        la = dill.load(inpt)
    return la


def test_lae(dataset, batch_size=1):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    latent_size = 2
    encoder = get_encoder(dataset, latent_size).eval().to(device)
    encoder.load_state_dict(torch.load(f"weights/{dataset}/encoder.pth"))

    la = load_laplace(f"weights/{dataset}/laplace_decoder.pkl")
    
    train_loader, val_loader = get_data(dataset, batch_size)

    # forward eval la
    x, z, labels, mu_rec, sigma_rec = [], [], [], [], []
    for i, (X, y) in tqdm(enumerate(val_loader)):
        X = X.view(X.size(0), -1).to(device)
        with torch.inference_mode():
            z += [encoder(X)]

            # pred_type : {glm, nn}
            # link_approx only relevant for classification
            pred_type =  "glm"
            mu, var = la(z[-1], pred_type = pred_type)

            x += [X.cpu()]
            labels += [y]
            mu_rec += [mu.detach().cpu()]
            sigma_rec += [var.sqrt().cpu()]

        # only show the first 50 points
        # if i > 50:
        #    break
    
    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z = torch.cat(z, dim=0).cpu().numpy()
    mu_rec = torch.cat(mu_rec, dim=0).numpy()
    sigma_rec = torch.cat(sigma_rec, dim=0).numpy()

    ###
    # Grid for probability map
    ###

    n_points_axis = 50
    x_min, x_max = z[:, 0].min(), z[:, 0].max() 
    y_min, y_max = z[:, 1].min(), z[:, 1].max() 
    zx_grid = np.linspace(x_min - 1.5, x_max + 1.5, n_points_axis, dtype=np.float32)
    zy_grid = np.linspace(y_min - 1.5, y_max + 1.5, n_points_axis, dtype=np.float32)

    xg_mesh, yg_mesh = np.meshgrid(zx_grid, zy_grid)
    xg = xg_mesh.reshape(n_points_axis ** 2, 1)
    yg = yg_mesh.reshape(n_points_axis ** 2, 1)
    Z_grid_test = np.hstack((xg, yg))
    Z_grid_test = torch.from_numpy(Z_grid_test).to(device)

    all_f_mu, all_f_sigma = [], []
    for i in tqdm(range(Z_grid_test.shape[0])):
        f_mu, f_var = la(Z_grid_test[i:i+1,:], pred_type = pred_type)

        all_f_mu += [f_mu.squeeze().detach().cpu()]
        all_f_sigma += [f_var.squeeze().sqrt().cpu()]

    f_mu = torch.stack(all_f_mu, dim=0)
    f_sigma = torch.stack(all_f_sigma, dim=0)

    # get diagonal elements
    idx = torch.arange(f_sigma.shape[1])
    sigma_vector = f_sigma.mean(axis=1) if pred_type == "nn" else f_sigma[:, idx, idx].mean(axis=1)

    # create figures
    if not os.path.isdir(f"figures/{dataset}"): os.makedirs(f"figures/{dataset}")

    plt.figure()
    if dataset == "mnist":
        for yi in np.unique(labels):
            idx = labels == yi
            plt.plot(z[idx, 0], z[idx, 1], 'x', ms=5.0, alpha=1.0)
    else:
        plt.plot(z[:, 0], z[:, 1], 'wx', ms=5.0, alpha=1.0)
    precision_grid = np.reshape(sigma_vector, (n_points_axis, n_points_axis))
    plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
    plt.colorbar()
    plt.savefig(f"figures/{dataset}/contour.png")
    plt.close(); plt.cla()

    if dataset == "mnist":
        for i in range(min(len(z), 10)):
            plt.figure()
            plt.subplot(1,3,1)

            plt.imshow(x[i].reshape(28,28))

            plt.subplot(1,3,2)
            plt.imshow(mu_rec[i].reshape(28,28))

            plt.subplot(1,3,3)
            N = 784 
            sigma = sigma_rec[i] if pred_type == "nn" else sigma_rec[i][np.arange(N), np.arange(N)]
            plt.imshow(sigma.reshape(28,28))

            plt.savefig(f"figures/{dataset}/recon_{i}.png")
            plt.close(); plt.cla()


def train_lae(dataset="mnist", n_epochs=50, batch_size=32):

    # initialize_model
    latent_size = 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(dataset, latent_size).eval().to(device)
    decoder = get_decoder(dataset, latent_size).eval().to(device)

    # load model weights
    encoder.load_state_dict(torch.load(f"weights/{dataset}/encoder.pth"))
    decoder.load_state_dict(torch.load(f"weights/{dataset}/decoder.pth"))

    train_loader, val_loader = get_data(dataset, batch_size)
    
    # create dataset
    z, x = [], []
    for X, y in tqdm(train_loader):
        X = X.view(X.size(0), -1).to(device)
        with torch.inference_mode():
            z += [encoder(X)]
            x += [X]
    
    z = torch.cat(z, dim=0).cpu()
    x = torch.cat(x, dim=0).cpu()

    z_loader = DataLoader(TensorDataset(z, x), batch_size=batch_size)

    # Laplace Approximation
    la = Laplace(decoder, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
    
    # Fitting
    la.fit(z_loader)

    la.optimize_prior_precision()
    # log_prior, log_sigma = torch.ones(1, requires_grad=True, device=device), torch.ones(1, requires_grad=True, device=device)
    # hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-2)
    # for i in range(n_epochs):
    #    hyper_optimizer.zero_grad()
    #    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    #    neg_marglik.backward()
    #    hyper_optimizer.step()

    # save weights
    save_laplace(la, f"weights/{dataset}/laplace_decoder.pkl")


if __name__ == "__main__":

    train = False
    dataset = "mnist"
    batch_size = 1

    # train or load laplace auto encoder
    if train:
        print("==> train lae")
        train_lae(
            dataset=dataset, 
            batch_size=batch_size
        )

    # evaluate laplace auto encoder
    print("==> evaluate lae")
    test_lae(dataset)