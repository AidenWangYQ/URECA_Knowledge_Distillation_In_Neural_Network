import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)  # Smaller network with fewer parameters
        self.fc2 = nn.Linear(300, 10)     # 10 classes for MNIST

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
