import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.student_model import StudentNet
from distillation import distillation_loss
from models.teacher_model import TeacherNet
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import argparse  # For command-line argument parsing

# Argument parser to accept experiment parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Train Student Model")
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[800, 800], help="Hidden layer sizes (e.g., 800 800)")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--temperature', type=float, default=5, help="Temperature for distillation")
    parser.add_argument('--use_distillation', type=bool, default=True, help="Whether to use distillation")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    return parser.parse_args()

# Train the student model
def train_student(args):
    # Data loading and transformation
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_model = StudentNet(
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout_rate,
        temperature=args.temperature,
        use_distillation=args.use_distillation
    ).to(device)

    teacher_model = TeacherNet().to(device) # Initialize the teacher model
    teacher_model.load_state_dict(torch.load('teacher_model.pth')) # Load pre-trained weights for the teacher model (into state_dict)
    teacher_model.eval()  # Set the teacher model to evaluation mode
    # Evaluation mode deisables dropoout & modifies batch normalization to ensure consistency.
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    # Set up TensorBoard writer
    writer = SummaryWriter('runs/student_experiment')  # Logs will be saved in this folder

    # Training loop
    for epoch in range(args.epochs):  # Train for the specified number of epochs
        student_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            student_outputs = student_model(images)  # Forward pass (student)
            teacher_outputs = teacher_model(images)  # Forward pass (teacher)

            # Compute distillation loss (if distillation is enabled)
            if args.use_distillation:
                loss = distillation_loss(student_outputs, teacher_outputs, labels, T=args.temperature)
            else:
                loss = torch.nn.CrossEntropyLoss()(student_outputs, labels)  # Standard cross-entropy loss
            
            loss.backward()  # Backpropagation
            optimizer.step()  # Update student model parameters

            running_loss += loss.item()

        # Log the training loss to TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch + 1)

        # Log histograms of weights after every epoch
        for name, param in student_model.named_parameters():
            if "weight" in name:  # Only log the weights (not gradients or biases)
                writer.add_histogram(f'{name}_weights', param, epoch + 1)

        # Log activations of the first layer
        activation = student_model.fc1(images.view(-1, 28*28))  # Get activations of the first layer
        writer.add_histogram('fc1/activation', activation, epoch + 1)

        # Log gradients of the weights (after backward pass)
        for name, param in student_model.named_parameters():
            if "weight" in name:  # Log gradients of the weights
                writer.add_histogram(f'{name}_gradients', param.grad, epoch + 1)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    # Save student model
    torch.save(student_model.state_dict(), 'student_model.pth')

    # Close the TensorBoard writer
    writer.close()

# Main entry point to start training
if __name__ == "__main__":
    args = parse_args()  # Parse experiment parameters
    train_student(args)  # Train with the given configurations
