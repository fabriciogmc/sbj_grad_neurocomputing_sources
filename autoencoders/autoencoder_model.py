""" A simple autoencoder model using pytorch and MLP

Author: Prof. Fabrício Galende Marques de Carvalho

"""

import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, subnet_topology):
        super().__init__()
        #output of this layer is the compressed input representation
        self.encoder = nn.Sequential(
            nn.Linear(input_size, subnet_topology[0]),
            nn.ReLU(),
            nn.Linear(subnet_topology[0], subnet_topology[1]),  
        )

        # output of this layer reconstructs the image from compressed
        # representation.
        self.decoder = nn.Sequential(
            nn.Linear(subnet_topology[1], subnet_topology[0]),
            nn.ReLU(),
            nn.Linear(subnet_topology[0], input_size),
            nn.Sigmoid()  # output normalization (0-1)
        )

    def forward(self, x):
        compressed = self.encoder(x)
        y = self.decoder(compressed)
        return y

    def encode(self, decoded):
        return self.encoder(decoded)

    def decode(self, encoded):
        return self.decoder(encoded)