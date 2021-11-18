#!/usr/bin/env python3
import torch
from torch.distributions import Distribution

def kl_by_sampling(p: Distribution, q: Distribution, n_samples: int = 1000):
    """
    Returns the mean KL by sampling.
    """
    x = p.rsample((n_samples,))  # (n_samples)x(output of p)
    kl = (p.log_prob(x) - q.log_prob(x)).mean(dim=0).abs()  # (output_of_p?)
    return kl