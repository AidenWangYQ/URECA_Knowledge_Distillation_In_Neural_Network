import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.teacher_model import TeacherNet
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import argparse  # For command-line argument parsing
from data_pipeline import get_data_loader  # Import the new data loading pipeline
from torchvision import datasets, transforms  # For test dataset loading

# Argument parser to accept experiment parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher Model")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--jitter', action='store_true', help="Whether to jitter images by 2 pixels")
    parser.add_argument('--omit_class', type=int, help="Class to omit from the dataset (e.g., 3 for omitting '3')")
    parser.add_argument('--classes', type=int, nargs='+', help="Classes to include (e.g., 7 8 for including only 7s and 8s)")
    parser.add_argument('--use_l2_regularization', action='store_true', help="Whether to apply L2 regularization")
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
    teacher_model = TeacherNet(
        hidden_sizes=[1200, 1200],  # Example hidden layer sizes
        dropout_rate=0.5,           # 50% dropout by default
        use_l2_regularization=args.use_l2_regularization
    ).to(device)  # Initialize the teacher model
    criterion = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for training
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)  # Adam optimizer

    # Set up TensorBoard writer
    writer = SummaryWriter('runs/teacher_experiment')  # Logs will be saved in this folder

    # Load the test dataset (unchanged by transformations from training)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Training loop
    for epoch in range(args.epochs):  # Train for the specified number of epochs
        teacher_model.train()  # Set model to training mode (affects layers like dropout, batch normalization)
        running_loss = 0.0
        for images, labels in train_loader:  # Iterate over batches in the training set
            images, labels = images.to(device), labels.to(device)  # Move to device

            optimizer.zero_grad()  # Zero the gradients
            outputs, l2_norm = teacher_model(images)  # Forward pass (teacher model)
            loss = criterion(outputs, labels)  # Compute loss between predicted and true values
            
            # If using L2 regularization, add the L2 penalty to the loss
            if args.use_l2_regularization:
                loss += 0.01 * l2_norm  # You can adjust the regularization strength (0.01 here)

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
        activation = teacher_model.fc_layers[0](images.view(-1, 28*28))  # Get activations of the first layer
        writer.add_histogram('fc1/activation', activation, epoch + 1)

        # Log gradients of the weights (after backward pass)
        for name, param in teacher_model.named_parameters():
            if "weight" in name:  # Log gradients of the weights
                writer.add_histogram(f'{name}_gradients', param.grad, epoch + 1)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

        # Calculate and log test errors after each epoch
        test_errors = evaluate_model(teacher_model, test_loader, device)
        writer.add_scalar('Test Errors', test_errors, epoch + 1)  # Log test errors to TensorBoard

        # Optionally, print test errors after each epoch
        print(f"Test Errors (Epoch {epoch + 1}): {test_errors}")

    # Save teacher model
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')  # Save the trained teacher model

    # Close the TensorBoard writer
    writer.close()

# Evaluation function to calculate test errors
def evaluate_model(model, test_loader, device):
    model.eval()  # Set to evaluation mode (disables dropout, batch normalization)
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move to device
            outputs, _ = model(images)  # Get the output from the model (ignore L2 norm)
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability

            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count the number of correct predictions

    # Calculate the number of test errors
    test_errors = total - correct
    return test_errors

# Main entry point to start training
if __name__ == "__main__":
    args = parse_args()  # Parse experiment parameters
    train_teacher(args)  # Train with the given configurations
