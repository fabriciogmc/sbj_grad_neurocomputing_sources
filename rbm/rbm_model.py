"""
Restricted Boltzmann Machine (RBM) implemented
with Pytorch

Author: Prof. Fabrício Galende Marques de Carvalho

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple RBM definition
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01) # Here we scale initial values to get closer to 0 ]-0.01, +0.01[
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # hidden layer bias
        self.v_bias = nn.Parameter(torch.zeros(n_visible)) # visible layer bias

    def sample_h(self, x):
        prob_h = torch.sigmoid(F.linear(x, self.W, self.h_bias))  # W*x + h_bias
        return prob_h, torch.bernoulli(prob_h) #outputs are hidden states probabilities and samples

    def sample_v(self, h, noise = 0.01):
        mu = F.linear(h, self.W.t(), self.v_bias)  # mu = gaussian average
        # In this case, gaussian probability is used because inputs are continuous variables.
        # normal sample: mu + noise
        v_sample = mu + torch.randn_like(mu) * noise  # small noise
        return mu, v_sample #outputs are visible states and visible states samples

    def forward(self, x):
        # In this case we return the sampled input (reconstruction)
        prob_h, h_sample = self.sample_h(x)
        prob_v, x_sample= self.sample_v(h_sample)
        return x_sample

    def contrastive_divergence(self, v0, eta=0.1):
        # Positive phase
        ph0, h0 = self.sample_h(v0) #h0 is a sample, ph0 is computed from input

        # Negative phase (reconstruction)
        pvk, vk = self.sample_v(h0)  #vk is a reconstruction from a sample
        phk, hk = self.sample_h(vk)  #phk is a hidden stated construction from a sample

        # Weight update using contrastive divergence
        # Division by v0.size(0) is needed because operation can be performed with an entire
        # batch of samples.  If a single point is used, v0.size(0) is not required. Note that
        # by default, when using tensorflow, v0.size(0) corresponds to batch size.
        # ph0: hidden states computed from inputs. v0: visible states
        # phk: hidden states computed from sampled inputs. vk: visible states computed from hidden states samples
        self.W.data += eta * (torch.matmul(ph0.t(), v0) - torch.matmul(phk.t(), vk)) / v0.size(0)

        # Mean along first dimension. In this case bias update is being performed in a way to
        # favor real measured states and computed hidden states probabilities. If no batch is used,
        # a single difference is computed.
        self.v_bias.data += eta * torch.mean(v0 - vk, dim=0) 
        self.h_bias.data += eta * torch.mean(ph0 - phk, dim=0)

        # Visible state reconstruction squared error (MSE)
        loss = torch.mean((v0 - vk) ** 2)
        return loss
