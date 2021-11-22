import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

def get_data(name, batch_size = 32):

    if name == "mnist":
        dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
        mnist_train, mnist_val = random_split(dataset, [55000, 5000])

        train_loader = DataLoader(mnist_train, batch_size=batch_size)
        val_loader = DataLoader(mnist_val, batch_size=batch_size)
    
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

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(X_val, y_test), batch_size=batch_size)

    else:
        raise NotImplemplenetError

    return train_loader, val_loader
