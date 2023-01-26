import torch
import torch.nn as nn




class SimpleCNN(torch.nn.Module):
    
    def __init__(self,use_cuda=False,pooling= False):
        super(SimpleCNN, self).__init__()
        self.use_cuda = use_cuda
        self.pooling = pooling
        self.conv_layer1 =  torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2)
        self.pool_layer1 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5,stride=2)
        self.pool_layer2 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        if pooling:
            self.fully_connected_layer = nn.Linear(64,64)
            self.final_layer = nn.Linear(64,11)
        else:
            self.fully_connected_layer = nn.Linear(1600, 64)
            self.final_layer = nn.Linear(64, 11)
    def forward(self,inp):
        """
        conv1 -> relu -> pool1 -> conv2 -> relu -> pool2 -> fc -> relu -> fc
        """
        x = torch.nn.functional.relu(self.conv_layer1(inp))
        if self.pooling:
            x = self.pool_layer1(x)
        x = torch.nn.functional.relu(self.conv_layer2(x))
        if self.pooling:
            x = self.pool_layer2(x)
        x = x.reshape(x.size(0),-1)
        x = torch.nn.functional.relu(self.fully_connected_layer(x))
        x = self.final_layer(x)
        return x

#• How many layers are there? Are they all convolutional? If not, what structure do they have?
# There are 2 convolutional layers, 2 pooling layers, 2 linear layers
# • Which activation function is used on the hidden nodes?
# ReLU is used on the hidden nodes.
#• What loss function is being used to train the network?
# CrossEntropyLoss is used to train the neural network.
#• How is the loss being minimized?
# The Adam optimizer is used to minimize the loss. We put the models parameters and the learning rate into the optimizer and it updates the parameters based on the loss.
