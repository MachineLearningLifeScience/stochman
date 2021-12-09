import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

def get_data(name, batch_size = 32):

    if name == "mnist":
        dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
        mnist_train, mnist_val = random_split(dataset, [55000, 5000])

        train_loader = DataLoader(mnist_train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(mnist_val, batch_size=batch_size, pin_memory=True)
    
    elif name == "swissrole":
        N_train = 50000
        N_val = 300

        # create simple sinusoid data set
        def swiss_roll_2d(noise=0.2, n_samples=100):
            z = 2.0 * np.pi * (1 + 2 * np.random.rand(n_samples))
            x = z * np.cos(z) + noise * np.random.randn(n_samples)
            y = z * np.sin(z) + noise * np.random.randn(n_samples)
            return torch.from_numpy(np.stack([x,y]).T.astype(np.float32)), torch.from_numpy(z.astype(np.float32))

        X_train, y_train = swiss_roll_2d(n_samples=N_train)
        X_val, y_test = swiss_roll_2d(n_samples=N_val)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(TensorDataset(X_val, y_test), batch_size=batch_size, pin_memory=True)

    else:
        raise NotImplemplenetError

    return train_loader, val_loader


def generate_latent_grid(x_min, x_max, y_min, y_max, n_points_axis=50, batch_size=1):

    x_margin = (x_max - x_min)*0.3
    y_margin = (x_max - x_min)*0.3
    zx_grid = np.linspace(x_min - x_margin, x_max + x_margin, n_points_axis, dtype=np.float32)
    zy_grid = np.linspace(y_min - y_margin, y_max + y_margin, n_points_axis, dtype=np.float32)

    xg_mesh, yg_mesh = np.meshgrid(zx_grid, zy_grid)
    xg = xg_mesh.reshape(n_points_axis ** 2, 1)
    yg = yg_mesh.reshape(n_points_axis ** 2, 1)
    Z_grid_test = np.hstack((xg, yg))
    Z_grid_test = torch.from_numpy(Z_grid_test)

    z_grid_loader = DataLoader(TensorDataset(Z_grid_test), batch_size=batch_size, pin_memory=True)

    return xg_mesh, yg_mesh, z_grid_loader    