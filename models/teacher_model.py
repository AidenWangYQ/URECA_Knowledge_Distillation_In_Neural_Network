import torch
import torch.nn as nn # Contains building blocks for neural networks, such as layers (nn.Linear) & loss functions
import torch.nn.functional as F # Provides functional API functions for common operations like activations (F.relu) that arn't layers themselves

class TeacherNet(nn.Module): # TeacherNet inherits from nn.Module (gains access to PyTorch's model-building features like automatic differentiation and parameter management)
    def __init__(self):
        super(TeacherNet, self).__init__()
        # Layer definitions
        # Linear Layers: Perform weighted sum of the inputs, adding a bias, to transform the input data into higher-dimensional space.
        self.fc1 = nn.Linear(28*28, 1200)  # Input to first hidden layer (Fully connected linear layer; Take input size 28*28, outputs tensor of size 1200)
        self.fc2 = nn.Linear(1200, 1200)   # Hidden to hidden layer (Second fully connected layer, take input of size 1200 and outputs 1200 values)
        self.fc3 = nn.Linear(1200, 10)     # Final output layer (for MNIST, 10 classes) (Final fully connected layer, which outputs 10 values, corresponding to the 10 class probabilities for MNIST(0-9))

    def forward(self, x): # Does foward pass of the model, which is how the input x flows through the network
        x = x.view(-1, 28*28)  # Flatten the image (to 784 elements)
        x = F.relu(self.fc1(x)) # Apply ReLU activation after 1st layer
        x = F.relu(self.fc2(x)) # Apply RElU activation after 2nd layer
        x = self.fc3(x) # Output layer (no activation function)
        return x

