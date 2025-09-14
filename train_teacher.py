# Code for training the teacher model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.teacher_model import TeacherNet
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = TeacherNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

# Set up TensorBoard writer
writer = SummaryWriter('runs/teacher_experiment_1')  # Logs will be saved in this folder

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

    # Log the training loss to TensorBoard
    writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch+1)

    # Log histograms of weights after every epoch
    for name, param in teacher_model.named_parameters():
        if "weight" in name:  # Only log the weights (not gradients or biases)
            writer.add_histogram(f'{name}_weights', param, epoch+1)

    # Log activations of the first layer (for teacher model)
    activation = teacher_model.fc1(images.view(-1, 28*28))  # Get activations of the first layer
    writer.add_histogram('fc1/activation', activation, epoch+1)

    # Log gradients of the weights (after backward pass)
    for name, param in teacher_model.named_parameters():
        if "weight" in name:  # Log gradients of the weights
            writer.add_histogram(f'{name}_gradients', param.grad, epoch+1)

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save teacher model
torch.save(teacher_model.state_dict(), 'teacher_model.pth')

# Teacher Model Evaluation: Evaluate the teacher model on the test set
teacher_model.eval()  # Set to evaluation mode

correct = 0
total = 0
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

with torch.no_grad():  # No need to compute gradients during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = teacher_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Teacher Model Test Accuracy: {accuracy:.2f}%')

# Close the TensorBoard writer
writer.close()
