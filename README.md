![Logo](stochman.png)
---

# StochMan - Stochastic Manifolds made easier

StochMan (Stochastic Manifolds) is a collection of elementary algorithms for computations 
on random manifolds learned from finite noisy data. Each algorithm assume that the considered 
manifold model implement a specific set of interfaces.

## Installation
As simple as
```
pip install stochman
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


### `stochman.geodesic`: computing geodesics made easy!


### `stochman.curves`: Simple curve objects







## Licence

Please observe the Apache 2.0 license that is listed in this repository. 

## BibTeX
If you want to cite the framework feel free to use this (but only if you loved it 😊):

```bibtex
@article{software:stochman,
  title={StochMan},
  author={Nicki S. Detlefsen and Dimitrios Kalatzis and Søren Hauberg},
  journal={GitHub. Note: https://github.com/MachineLearningLifeScience/stochman/},
  year={2021}
}
```







