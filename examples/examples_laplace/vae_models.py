# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:55:54 2018

@author: nsde
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math


def get_encoder(dataset, latent_size=2):

    if dataset == "mnist":
        encoder = Encoder_mnist(latent_size)
    elif dataset == "swissrole":
        encoder = Encoder_swissrole(latent_size)
    else:
        raise NotImplemplenetError

    return encoder


class Encoder_swissrole(nn.Module):
    def __init__(self, latent_size):
        super(Encoder_swissrole, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh()
        )

        self.mu_head = nn.Sequential(
            nn.Linear(50, latent_size)
        )

        self.log_var_head = nn.Sequential(
            nn.Linear(50, latent_size)
        )

    def forward(self, x):

        embed = self.encoder(x)
        mu = self.mu_head(embed)
        log_var = self.log_var_head(embed)

        return mu, log_var


class Encoder_mnist(nn.Module):
    def __init__(self, latent_size):
        super(Encoder_mnist, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.mu_head = nn.Sequential(
            nn.Linear(256, latent_size)
        )

        self.log_var_head = nn.Sequential(
            nn.Linear(256, latent_size)
        )
        
    def forward(self, x):

        embed = self.encoder(x)
        mu = self.mu_head(embed)
        log_var = self.log_var_head(embed)

        return mu, log_var
