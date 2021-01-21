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

Stoch includes a number of modules that each defines a set of functionalities for
working with manifold data

# `stochman.nnj`: `torch.nn` with jacobians

Many times it is requirement that working with the jacobian of the different operations.
`stochman.nnj` provides plug-in replacements for the most used `torch.nn` layers such
as `Linear`, `BatchNorm1d` ect. and commonly used activation functions such as `ReLU`,
`Sigmoid` ect. Working with 

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

Implementing your own layers requires simply:
* In addition to defining `forward`, the method called `_jacobian` should also be implemented.
* If you implement an activation function it should also inheret from `nnj.ActivationJacobian`


# stochman.manifold





## Licence

Please observe the Apache 2.0 license that is listed in this repository. 

## BibTeX
If you want to cite the framework feel free to use this (but only if you loved it 😊):

```bibtex
@article{detlefsen2021stochman,
  title={StochMan},
  author={Detlefsen, Nicki S. et al.},
  journal={GitHub. Note: https://github.com/CenterBioML/stochman},
  year={2021}
}
```







