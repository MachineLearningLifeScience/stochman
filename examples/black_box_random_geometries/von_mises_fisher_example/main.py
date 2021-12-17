"""
In this example we define a StatisticalManifold using a
custom-made distribution in PyTorch: a von Mises-Fisher
(which is available in vmf.py)

This script registers a KL divergence in Torch, which
is a necessary step for the StatisticalManifold to work.
The way this KL divergence is computed is by estimating it
with Monte Carlo samples:

from torch.distributions.kl import register_kl

@register_kl(VonMisesFisher, VonMisesFisher)
def _kl_vmf(p: VonMisesFisher, q: VonMisesFisher, n_samples: int = 100):
    '''
    Computes the KL divergence between two vMF distributions
    by sampling.
    '''
    x = p.rsample(n_samples)
    kl = (p.log_prob(x) - q.log_prob(x)).mean(dim=0).abs()

    return kl
"""

from .vmf import VonMisesFisher
