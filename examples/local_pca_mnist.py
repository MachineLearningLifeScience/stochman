#!/usr/bin/env python3
import torch
import numpy as np
from torchvision.datasets import MNIST

import stochman
import torchplot as plt


def get_subset_mnist(n: int = 1000):
    dataset = MNIST(root='', download=True)
    N = dataset.data.shape[0]
    idx = np.random.choice(np.arange(N), size=n)
    return dataset.data[idx], dataset.targets[idx]

# Read data
data, targets = get_subset_mnist(n=1000)
data = data.reshape(data.shape[0], -1)
N, D = data.shape
    
# Parameters for metric
sigma = 0.1
rho = 0.1

# Create metric
M = stochman.manifold.LocalVarMetric(data=data, sigma=sigma, rho=rho)

## Plot metric and data
ran = torch.linspace(-2.5, 2.5, 100)
X, Y = torch.meshgrid([ran, ran])
XY = torch.stack((X.flatten(), Y.flatten()), dim=1) # 10000x2
gridM = M.metric(XY) # 10000x2
Mim = gridM.sum(dim=1).reshape((100, 100)).detach().numpy().T
plt.imshow(Mim, extent=(ran[0], ran[-1], ran[0], ran[-1]), origin='lower')
plt.plot(data[:, 0].numpy(), data[:, 1].numpy(), 'w.', markersize=1)

## Compute geodesics in parallel
p0 = data[torch.randint(high=N, size=[10], dtype=torch.long)] # 10xD
p1 = data[torch.randint(high=N, size=[10], dtype=torch.long)] # 10xD
C, success = M.connecting_geodesic(p0, p1)
C.plot()
C.constant_speed(M)
C.plot()
plt.show()

## Compute shooting geodesic as a sanity check
p0 = data[0] # 1xD
p1 = data[1] # 1xD
C, success = M.connecting_geodesic(p0, p1)
C.plot()

# p = C.begin
# with torch.no_grad(): 
#     v = C.deriv(torch.zeros(1))
#     c, dc = shooting_geodesic(M, p, v, t=torch.linspace(0, 1, 100))
# plt.plot(c[:,0,0], c[:,1, 0], 'o')
