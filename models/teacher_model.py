import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherNet(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[1200, 1200], output_size=10, dropout_rate=0.5, use_l2_regularization=False):
        super(TeacherNet, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.use_l2_regularization = use_l2_regularization  # Whether to apply L2 regularization
        self.dropout_rate = dropout_rate  # Dropout rate (50% dropout by default)

        # Create a list of layers (input to hidden layers)
        self.fc_layers = nn.ModuleList()  # List to hold the layers
        prev_size = input_size  # Start with the input size (784 for MNIST)
        for size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, size))  # Create a Linear layer
            prev_size = size  # Update prev_size to the current layer size for the next layer
        
        # Output layer
        self.fc_out = nn.Linear(prev_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)  # Dropout with the specified probability

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input image (28x28 -> 784)

        # Pass through hidden layers with ReLU activation and dropout
        for fc in self.fc_layers:
            x = F.relu(fc(x))  # Apply ReLU activation after each hidden layer
            x = self.dropout(x)  # Apply dropout after each hidden layer
        
        x = self.fc_out(x)  # Output layer (no activation function here)
        
        # L2 Regularization (apply L2 penalty if enabled)
        if self.use_l2_regularization:
            l2_norm = sum(torch.sum(param**2) for param in self.parameters())  # Sum of squared parameters for L2 regularization
            return x, l2_norm  # Return the L2 penalty along with the output

        return x  # Return the raw output if L2 regularization is not used

