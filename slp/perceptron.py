import numpy as np


class ActivationFunction:

    def __init__(self, act_function):
        self.act_function = act_function

    def activate(self,v):
        return self.act_function(v)


class Perceptron:

    def __init__(self, act_function, n_inputs, n_outputs):
        self.act_function = act_function
        self.n_inputs = n_inputs  # no bias is taken into account
        self.n_outputs = n_outputs
        self.synaptic_weights = np.random.rand(n_outputs, n_inputs)

    def output(self, inputs):
        v = np.dot(self.synaptic_weights, np.array(inputs))
        y_e = list( map(self.act_function.activate, v) )
        return y_e
        
    def learn(self, y_t, y_e, x, alpha):
        for i in range(self.n_outputs):
            for j in range(self.n_inputs):
                error = y_t[i] - y_e[i]
                self.synaptic_weights[i,j] = self.synaptic_weights[i,j] + alpha * error * x[j]
    
    
                
                
        

        
