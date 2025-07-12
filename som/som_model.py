"""
Simple Self Organizing Map Network model
Author: Prof. Fabrício Galende Marques de Carvalho
"""

import numpy as np


class SOMNeuron:
    def __init__(self, dimension, weights = None, post_activation = None):

        "Weights initialization"
        if not weights:
            self.weights = np.random.normal(loc=0, scale=1, size=(dimension, 1)) ## mean 0, standar deviation = 1 
            norm = np.linalg.norm(self.weights) #vector norm
            self.weights = self.weights/norm    #here we normalize weights to make learning more efficient
        else:
            self.weights = weights

        "Neuron condition initialization, if it is not given"
        if not post_activation:
            self.post_activation = np.zeros((1,1))
        else:
            self.post_activation = post_activation


class SOMNetwork:

    def __init__(self, grid_dimensions, input_dimension, neurons = None):
        "Neurons initialization"
        if not neurons:
            self.grid_dimensions = grid_dimensions
            self.neurons = np.empty((grid_dimensions[0], grid_dimensions[1]), dtype=object)  
            self.post_activation = np.zeros((grid_dimensions[0], grid_dimensions[1]))
            for i in range(0, self.grid_dimensions[0]):
                for j in range(0,self.grid_dimensions[1]):
                    self.neurons[i,j] = SOMNeuron(input_dimension)
        else:
            self.grid_dimensions = [len(neurons), len(neurons[1])]
            self.neurons = neurons
            for i in range(0, self.grid_dimensions[0]):
                for j in range(0,self.grid_dimensions[1]):
                    self.post_activation = self.neurons[i,j].post_activation

    def organize(self, x, eta, neighbor_fcn, sigma):
        """
        This method organize the SOM Network based on a
        single exaple. Complete learning shall be performed
        using the complete learning set.     
        The learning rule is given by:
        neuron.weights[i,j](k) = neuron.weights[i,j](k-1) + eta*h(i,j)*(x-neuron.weights[i,j])
        """

        """
        The following operation could be done using 
        numpy tensor operation, but it was performed
        using a traditional loop for learning purposes.
        Moreover, it is supposed that imput vectors are
        normalized with respect to its norm (= 1)
        """
        for i in range(self.grid_dimensions[0]):
            for j in range(self.grid_dimensions[1]):
                self.neurons[i,j].post_activation = np.dot(x.flatten(), self.neurons[i,j].weights.flatten())
                self.post_activation[i,j] = self.neurons[i,j].post_activation
            
        # Now we determine the winning neuron
        winning_neuron_max_flat_idx= np.argmax(self.post_activation)
        wnx , wny =  np.unravel_index(winning_neuron_max_flat_idx, self.neurons.shape)
    
        #finally we perform weight adjustment based upon 
        # neighbor function and learning rate that were given

        for i in range(self.grid_dimensions[0]):
            for j in range(self.grid_dimensions[1]):
                self.neurons[i,j].weights = self.neurons[i,j].weights + neighbor_fcn(wnx, wny, i,j,sigma)*eta*(x - self.neurons[i,j].weights)

        

        
        
            
            
        
            

        
        