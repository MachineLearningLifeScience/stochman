![Logo](images/stochman.png)
---

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/stochman)](https://pypi.org/project/stochman/)
[![PyPI Status](https://badge.fury.io/py/stochman.svg)](https://badge.fury.io/py/stochman)
[![PyPI Status](https://pepy.tech/badge/stochman)](https://pepy.tech/badge/stochman)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/MachineLearningLifeScience/stochman/blob/master/LICENSE)
[![Tests](https://github.com/MachineLearningLifeScience/stochman/actions/workflows/tests.yml/badge.svg)](https://github.com/MachineLearningLifeScience/stochman/blob/master/.github/workflows/tests.yml)
[![codecov](https://codecov.io/gh/MachineLearningLifeScience/stochman/branch/master/graph/badge.svg)](https://codecov.io/gh/MachineLearningLifeScience/stochman)
# StochMan - Stochastic Manifolds made easier

StochMan (Stochastic Manifolds) is a collection of elementary algorithms for computations 
on random manifolds learned from finite noisy data. Each algorithm assume that the considered 
manifold model implement a specific set of interfaces.

## Installation

For the latest release
```bash
pip install stochman
```
For master version with most recent changes we recommend:
```bash
git clone https://github.com/MachineLearningLifeScience/stochman
cd stochman
python setup.py install
```

## API overview

`StochMan` includes a number of modules that each defines a set of functionalities for
working with manifold data.

### `stochman.nnj`: `torch.nn` with jacobians

Key to working with Riemannian geometry is the ability to compute jacobians. The jacobian matrix
contains the first order partial derivatives. `stochman.nnj` provides plug-in replacements for the many 
used `torch.nn` layers such as `Linear`, `BatchNorm1d` etc. and commonly used activation functions such as `ReLU`,
`Sigmoid` etc. that enables fast computations of jacobians between the input to the layer and the output. 

``` python
import torch
from stochman import nnj

model = nnj.Sequential(nnj.Linear(10, 5),
                       nnj.ReLU())
x = torch.randn(100, 10)
y, J = model(x, jacobian=True)
print(y.shape) # output from model: torch.size([100, 5])
print(J.shape) # jacobian between input and output: torch.size([100, 5, 10])
```

### `stochman.manifold`: Interface for working with Riemannian manifolds

A manifold can be constructed simply by specifying its metric. The example below shows a toy example where the metric grows with the distance to the origin.

``` python
import torch
from stochman.manifold import Manifold

class MyManifold(Manifold):
    def metric(self, c, return_deriv=False):
        N, D = c.shape  # N is number of points where we evaluate the metric; D is the manifold dimension
        sq_dist_to_origin = torch.sum(c**2, dim=1, keepdim=True)  # Nx1
        G = (1 + sq_dist_to_origin).unsqueeze(-1) * torch.eye(D).repeat(N, 1, 1)  # NxDxD
        return G
        
model = MyManifold()
p0, p1 = torch.randn(1, 2), torch.randn(1, 2)
c, _ = model.connecting_geodesic(p0, p1)  # geodesic between two random points
```

If you manifold is embedded (e.g. an autoencoder) then you only have to provide a function for realizing the embedding (i.e. a decoder) and StochMan takes care of the rest (you, however, have to learn the autoencoder yourself).

``` python
import torch
from stochman.manifold import EmbeddedManifold

class Autoencoder(EmbeddedManifold):
    def embed(self, c, jacobian = False):
        return self.decode(c)
        
model = Autoencoder()
p0, p1 = torch.randn(1, 2), torch.randn(1, 2)
c, _ = model.connecting_geodesic(p0, p1)  # geodesic between two random points
```

### `stochman.geodesic`: computing geodesics made easy!

Geodesics are energy-minimizing curves, and StochMan computes them as such. You can use the high-level `Manifold` interface or the more explicit one:

``` python
import torch
from stochman.geodesic import geodesic_minimizing_energy
from stochman.curves import CubicSpline

model = MyManifold()
p0, p1 = torch.randn(1, 2), torch.randn(1, 2)
curve = CubicSpline(p0, p1)
geodesic_minimizing_energy(curve, model)
```

### `stochman.curves`: Simple curve objects

We often want to manipulate curves when computing geodesics. StochMan provides an implementation of cubic splines and discrete curves, both with the end-points fixed.

``` python
import torch
from stochman.curves import CubicSpline

p0, p1 = torch.randn(1, 2), torch.randn(1, 2)
curve = CubicSpline(p0, p1)

t = torch.linspace(0, 1, 50)
ct = curve(t)  # 50x2
```

## Licence

Please observe the Apache 2.0 license that is listed in this repository. 

## BibTeX
If you want to cite the framework feel free to use this (but only if you loved it 😊):

```bibtex
@article{software:stochman,
  title={StochMan},
  author={Nicki S. Detlefsen and Alison Pouplin and Cilie W. Feldager and Cong Geng and Dimitris Kalatzis and Helene Hauschultz and Miguel González Duque and Frederik Warburg and Marco Miani and Søren Hauberg},
  journal={GitHub. Note: https://github.com/MachineLearningLifeScience/stochman/},
  year={2021}
}
```
