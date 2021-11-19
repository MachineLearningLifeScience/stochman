#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from vae_normal import VAE_normal
from torchvision.datasets import MNIST

load = False
filename = '../models/vae_normal'

mnist = MNIST('../data/', download=True)
X = mnist.train_data.reshape(-1, 784).numpy() / 255.0
y = mnist.train_labels.numpy()

S = VAE_normal(2)

# if load:
#     S.load(filename)
# else:
S.fit(X, n_epochs=2, learning_rate=1e-3, batch_size=64, verbose=True, labels=y)