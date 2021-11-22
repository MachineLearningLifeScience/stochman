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

def get_decoder(dataset, latent_size=2):

    if dataset == "mnist":
        decoder = Decoder_mnist(latent_size)
    elif dataset == "swissrole":
        decoder = Decoder_swissrole(latent_size)
    else:
        raise NotImplemplenetError

    return decoder

class Encoder_swissrole(nn.Module):
    def __init__(self, latent_size):
        super(Encoder_swissrole, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(nn.Linear(2, 50),
                                    nn.Tanh(),
                                    nn.Linear(50, latent_size))

    def forward(self, x):
        return self.encoder(x)


class Decoder_swissrole(nn.Module):
    def __init__(self, latent_size, device="cpu", learning_rate=1e-4):
        super(Decoder_swissrole, self).__init__()
        self.latent_size = latent_size
        self.decoder = nn.Sequential(nn.Linear(latent_size, 50),
                                    nn.Tanh(),
                                    nn.Linear(50, 2))
    def forward(self, x):
        return self.decoder(x)

      
class Encoder_mnist(nn.Module):
    def __init__(self, latent_size):
        super(Encoder_mnist, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size)
        )
        
    def forward(self, x):
        return self.encoder(x)


class Decoder_mnist(nn.Module):
    def __init__(self, latent_size):
        super(Decoder_mnist, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )
    
    def forward(self, x):
        return self.decoder(x)

