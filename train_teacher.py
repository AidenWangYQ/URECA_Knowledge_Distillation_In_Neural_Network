# Code for training the teacher model
import torch # PyTorch library for deep learning tasks
import torch.optim as optim # Contains optimization algorithms like Adam
from torch.utils.data import DataLoader # Loads data in batches to facilitate training
from torchvision import datasets, transforms # Tools for loading and transforming datasets
from models.teacher_model import TeacherNet # A custom model, presumably a neural network for the teacher model (from our models/teacher_model.py)
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard, a utility to log data for TensorBoard visualization

# Data loading
transform = transforms.Compose([transforms.ToTensor()]) # Converts the images into PyTorch tensors (required for processing in neural networks)
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform) # Download the MNIST data set if not already present in 'data' directory; loads the training part of the dataset; applies transformation to each image
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Loads the dataset in batches; batch size of 64; Shuffles data each time to avoid biasing the model with data order 

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Check if CUDA(GPU) is available & sets the device to wither CUDA(GPU) or 'cpu'
teacher_model = TeacherNet().to(device) # Instantiates the TeacherNet model
criterion = torch.nn.CrossEntropyLoss() # Defines the loss function as cross-entropy loss, which is common for classification tasks
optimizer = optim.Adam(teacher_model.parameters(), lr=0.001) # Sets up the Adam optimizer to update the model's parametes during training with a learning rate of 0.001 (To minimise the loss function)

# Set up TensorBoard writer
writer = SummaryWriter('runs/teacher_experiment_1')  # Logs will be saved in this folder; Sets up the SummaryWriter to the directory stated

# Training loop
for epoch in range(5):  # Train for 5 epochs (can increase)
    teacher_model.train()  # Set model to training mode (affects layers like dropout, batch normalization)
    running_loss = 0.0 
    for images, labels in train_loader: # Labels are hard targets, images are data
        images, labels = images.to(device), labels.to(device) # Images & labels are moved to selected device 

        optimizer.zero_grad()  # Zero the parameter gradients (Clears the previous gradients)
        outputs = teacher_model(images)  # Forward pass (input data is passed throguh neural network, processed layer by layer to produce an output: predicted class probabilities)
        loss = criterion(outputs, labels)  # Calculate loss between predicted & true values
        loss.backward()  # Performs backpropagation, computing the gradients
        # Backpropagation is used to calculate gradients of loss function wrt to the model's parameters (weights & biases); These gradients indicate how to adjust the parameters to reduce the loss
        # Gradients are values that indicate how much change in each model parameter (e.g. weights) will impact the loss. Gradients are computed during back propagation
        optimizer.step()  # Optimize the weights (Updates the model paratmeters based on the computed gradients)
        # Purpose of updating weights based on gradients: Optimise model, reducing loss & improving performance
        running_loss += loss.item() # Keep track of the total loss for the current epoch

    # Log the training loss to TensorBoard
    writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch+1) # Average loss per unit of data

    # Log histograms of weights after every epoch
    for name, param in teacher_model.named_parameters():
        if "weight" in name:  # Only log the weights (not gradients or biases)
            writer.add_histogram(f'{name}_weights', param, epoch+1)

    # Log activations of the first layer (for teacher model)
    activation = teacher_model.fc1(images.view(-1, 28*28))  # Get activations of the first layer (show us how much input data is transformed as it moves through the model)
    writer.add_histogram('fc1/activation', activation, epoch+1)

    # Log gradients of the weights (after backward pass)
    for name, param in teacher_model.named_parameters():
        if "weight" in name:  # Log gradients of the weights
            writer.add_histogram(f'{name}_gradients', param.grad, epoch+1)

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save teacher model
torch.save(teacher_model.state_dict(), 'teacher_model.pth') # Saves the model into 'teacher_model.pth'

# Teacher Model Evaluation: Evaluate the teacher model on the test set
teacher_model.eval()  # Set to evaluation mode (disabling layers like dropout and batch normalization)

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
