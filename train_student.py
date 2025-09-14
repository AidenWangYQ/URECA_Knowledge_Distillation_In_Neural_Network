# Code for training the student model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.student_model import StudentNet
from distillation import distillation_loss
from models.teacher_model import TeacherNet

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
student_model = StudentNet().to(device)
teacher_model = TeacherNet().to(device)
teacher_model.load_state_dict(torch.load('teacher_model.pth'))
teacher_model.eval()  # Set the teacher model to evaluation mode
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Train for 5 epochs
    student_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        student_outputs = student_model(images)  # Forward pass (student)
        teacher_outputs = teacher_model(images)  # Forward pass (teacher)

        # Compute distillation loss
        loss = distillation_loss(student_outputs, teacher_outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize student model

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save student model
torch.save(student_model.state_dict(), 'student_model.pth')
