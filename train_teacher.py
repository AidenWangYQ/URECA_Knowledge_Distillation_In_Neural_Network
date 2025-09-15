import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.teacher_model import TeacherNet
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import argparse  # For command-line argument parsing
from data_pipeline import get_data_loader  # Import the new data loading pipeline

# Argument parser to accept experiment parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher Model")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--jitter', action='store_true', help="Whether to jitter images by 2 pixels")
    parser.add_argument('--omit_class', type=int, help="Class to omit from the dataset (e.g., 3 for omitting '3')")
    parser.add_argument('--classes', type=int, nargs='+', help="Classes to include (e.g., 7 8 for including only 7s and 8s)")
    return parser.parse_args()

# Train the teacher model
def train_teacher(args):
    # Get the appropriate data loader based on the experiment parameters
    train_loader = get_data_loader(
        classes=args.classes,            # Classes to include (e.g., 7s and 8s)
        omit_class=args.omit_class,      # Class to omit (e.g., 3)
        jitter=args.jitter,              # Whether to jitter the images
        batch_size=args.batch_size       # Batch size
    )

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = TeacherNet().to(device)  # Initialize the teacher model
    criterion = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for training
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)  # Adam optimizer

    # Set up TensorBoard writer
    writer = SummaryWriter('runs/teacher_experiment')  # Logs will be saved in this folder

    # Training loop
    for epoch in range(args.epochs):  # Train for the specified number of epochs
        teacher_model.train()  # Set model to training mode (affects layers like dropout, batch normalization)
        running_loss = 0.0
        for images, labels in train_loader:  # Iterate over batches in the training set
            images, labels = images.to(device), labels.to(device)  # Move to device (GPU or CPU)

            optimizer.zero_grad()  # Zero the gradients
            outputs = teacher_model(images)  # Forward pass (teacher model)
            loss = criterion(outputs, labels)  # Compute loss between predicted and true values
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the teacher model's parameters

            running_loss += loss.item()

        # Log the training loss to TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch + 1)

        # Log histograms of weights after every epoch
        for name, param in teacher_model.named_parameters():
            if "weight" in name:  # Only log the weights (not gradients or biases)
                writer.add_histogram(f'{name}_weights', param, epoch + 1)

        # Log activations of the first layer (for teacher model)
        activation = teacher_model.fc1(images.view(-1, 28*28))  # Get activations of the first layer
        writer.add_histogram('fc1/activation', activation, epoch + 1)

        # Log gradients of the weights (after backward pass)
        for name, param in teacher_model.named_parameters():
            if "weight" in name:  # Log gradients of the weights
                writer.add_histogram(f'{name}_gradients', param.grad, epoch + 1)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    # Save teacher model
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')  # Save the trained teacher model

    # Close the TensorBoard writer
    writer.close()

# Main entry point to start training
if __name__ == "__main__":
    args = parse_args()  # Parse experiment parameters
    train_teacher(args)  # Train with the given configurations
