import numpy as np
import scipy.special


# definition of neural network
class neuralNetwork:

    # initializator of nn
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # give count of nodes for input, hidden and output slices
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # coefficient of trainig
        self.lr = learningrate
        # matrix of bond coefficients wih(between input and hidden) & who (between output and hidden)
        # weight coefficient bonds between nodes i and nodes j next slice defined as w_i_j:
        # w11, w22, w12, w22 and etc.

        # standard way:
        # self.wih = (np.random.rand(self.hnodes, self.hnodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        # advanced way of definition matrix of weight coefficients
        # we take normal distribution with center in 0 and scale of 1/sqrt(node)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # using of sigmoid like activation function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # train of nn
    def train(self):
        pass

    # questioner of nn
    def query(self, inputs_list):
        # change input massive into 2dim massive
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate input signals for hidden slice
        # scalar matrix multiplication
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate output signals for hidden slice
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate input signals for output slice
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate output signals for output slice
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# main code
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

# example of nn
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

np.random.rand(3, 3) - 0.5
