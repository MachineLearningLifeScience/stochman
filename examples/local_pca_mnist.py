#!/usr/bin/env python3
import numpy as np
import torch
import torchplot as plt
from torchvision.datasets import MNIST

import stochman


def get_subset_mnist(n: int = 1000):
    dataset = MNIST(root="", download=True)
    data = dataset.data[dataset.targets == 1]
    N = data.shape[0]
    idx = np.random.choice(np.arange(N), size=n)
    return data[idx]


# Read data
data = get_subset_mnist(n=1000)
data = data.reshape(data.shape[0], -1).to(torch.float)
cov = torch.cov(data.t())
values, vectors = torch.linalg.eigh(cov)
proj = vectors[:, -2:] / values[-2:].sqrt().unsqueeze(0)
data = data @ proj
N, D = data.shape

# Parameters for metric
sigma = 0.1
rho = 0.1

# Create metric
M = stochman.manifold.LocalVarMetric(data=data, sigma=sigma, rho=rho)

# Plot metric and data
plt.figure()
ran = torch.linspace(-3.0, 3.0, 100)
X, Y = torch.meshgrid([ran, ran], indexing="ij")
XY = torch.stack((X.flatten(), Y.flatten()), dim=1)  # 10000x2
gridM = M.metric(XY)  # 10000x2
Mim = gridM.sum(dim=1).reshape((100, 100)).detach().t()
plt.imshow(Mim, extent=(ran[0], ran[-1], ran[0], ran[-1]), origin="lower")
plt.plot(data[:, 0], data[:, 1], "w.", markersize=1)

# Compute geodesics in parallel
p0 = data[torch.randint(high=N, size=[10], dtype=torch.long)]  # 10xD
p1 = data[torch.randint(high=N, size=[10], dtype=torch.long)]  # 10xD
C, success = M.connecting_geodesic(p0, p1)
C.plot()
C.constant_speed(M)
C.plot()

# Construct discretized manifold
DM = stochman.discretized_manifold.DiscretizedManifold()
DM.fit(M, [ran, ran], batch_size=100)

# Compute discretized geodesics
plt.figure()
ran2 = torch.linspace(-3.0, 3.0, 133)
X2, Y2 = torch.meshgrid([ran2, ran2], indexing="ij")
XY2 = torch.stack((X2.flatten(), Y2.flatten()), dim=1)  # 10000x2
DMim = DM.metric(XY2).log().sum(dim=1).view(133, 133).t()
plt.imshow(DMim, extent=(ran[0], ran[-1], ran[0], ran[-1]), origin="lower")
plt.plot(data[:, 0], data[:, 1], "w.", markersize=1)
p0 = data[torch.randint(high=N, size=[10], dtype=torch.long)]  # 10xD
p1 = data[torch.randint(high=N, size=[10], dtype=torch.long)]  # 10xD
C, success = DM.connecting_geodesic(p0, p1)
C.plot()

t = torch.linspace(0, 1, 100)
with torch.no_grad():
    print(DM.curve_length(C(t)))
    print(DM.dist2(p0, p1).sqrt())

# p = C.begin
# with torch.no_grad():
#     v = C.deriv(torch.zeros(1))
#     c, dc = shooting_geodesic(M, p, v, t=torch.linspace(0, 1, 100))
# plt.plot(c[:,0,0], c[:,1, 0], 'o')


plt.show()
