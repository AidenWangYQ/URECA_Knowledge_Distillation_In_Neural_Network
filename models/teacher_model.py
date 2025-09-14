import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 1200)  # Input to first hidden layer
        self.fc2 = nn.Linear(1200, 1200)   # Hidden to hidden layer
        self.fc3 = nn.Linear(1200, 10)     # Final output layer (for MNIST, 10 classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
