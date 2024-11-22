import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, width, input_dim, output_dim, path=None, checkpoint=1):
        super(DQN, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._width = width

        self.model_definition()

        if path is not None:
            print(path)
            # load the model from a saved checkpoint
            self.layers = torch.load(path + str(checkpoint))

    def model_definition(self):

        """
        Define the neural network for the DQN agent.
        """
        
        # TODO : Define the neural network structure here.
        # Define the model and assign it to self.layers
        # ex: self.layers = Model() in which Model(), you need to define as per the instructions below.
 
        # In our default network, we have one input layer with self._input_dim number of neurons.
        # We have 4 hidden layers with self._width number of neurons each.
        # We have one output layer with self._output_dim number of neurons.
        # Note that self._output_dim = 4 as it depicts the number of actions we have.
        # Each layer is followed by a ReLu non linearity except the output layer.
        # self._width, self._output_dim, self._input_dim are defined in training_settings.ini.
        
        self.layers = nn.Sequential(
            nn.Linear(self._input_dim, self._width),
            # nn.ReLU(),
            nn.Mish(),
            nn.Linear(self._width, self._width),
            # nn.ReLU(),
            nn.Mish(),
            nn.Linear(self._width, self._width),
            # nn.ReLU(),
            nn.Mish(),
            nn.Linear(self._width, self._width),
            # nn.ReLU(),
            nn.Mish(),
            nn.Linear(self._width, self._output_dim)
        )
            
        # Following documentation will be helpful in this task.
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    def forward(self, x):
        """
        Execute the forward pass through the neural network
        """
        # TODO : Define the forward pass of the neural network where x is the input tensor to the network.
        
        # Inputs: x defines the observation of the agent : x is a vector of size 80 where each element of the
        # vector corresponds to a cell in the state.
        
        # Retun: return the output of the neural network (a tensor of 4 values)
        return self.layers(x)

