# Code for training the teacher model

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.teacher_model import TeacherNet

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = TeacherNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Train for 5 epochs (can increase)
    teacher_model.train()  # Set model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = teacher_model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize the weights
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save teacher model
torch.save(teacher_model.state_dict(), 'teacher_model.pth')
