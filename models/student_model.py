import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamic student class that can create any number of layers and decide regularization methods depending on input
class StudentNet(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[800, 800], output_size=10, dropout_rate=0.0, temperature=1, use_distillation=False):
        super(StudentNet, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.use_distillation = use_distillation  # Whether or not to use distillation
        self.temperature = temperature  # Temperature for distillation
        
        # Dynamically create hidden layers based on hidden_sizes list
        self.fc_layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, size))
            prev_size = size
        
        self.fc_out = nn.Linear(prev_size, output_size)
        
        # Dropout as a regularization technique
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input image
        
        # Pass through hidden layers with ReLU activation and dropout (if enabled)
        for fc in self.fc_layers:
            x = F.relu(fc(x))  # Apply ReLU activation
            x = self.dropout(x)  # Apply dropout after each hidden layer
        
        x = self.fc_out(x)  # Output layer (no activation function here)
        
        # If using distillation, apply temperature scaling (this is for distillation loss)
        if self.use_distillation:
            x = x / self.temperature
        
        return x

