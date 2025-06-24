# Single Neuron Model Using Pytorch
# Prof. Fabrício Galende Marques de Carvalho
#
# To install Pytorch, activate your virtual environment and type
# pip3 install torch, or if you want to install all the libraries,
# including Jupyter Notebook, type 
# pip3 install -r requirements.txt

import torch
import torch.nn as nn

class Neuron:
    def __init__(self, weights, bias, act_fcn):
        n_features = len(weights)
        self.neuron = nn.Linear(in_features=n_features, out_features=1, bias=True) #dense neuron
        with torch.no_grad(): #no gradient is computed or registered for future use.
            self.neuron.weight[:] = torch.tensor([weights])
            self.neuron.bias[:] = torch.tensor([bias])
        self.act_fcn = act_fcn
        self.pre_act = 0

    def output(self, x):
        self.pre_act = self.neuron(x)
        return self.act_fcn(self.neuron(x))
    
