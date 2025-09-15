import torch
from torchvision import datasets, transforms

# Custom Dataset for filtering specific classes
class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes=[7, 8]):
        self.dataset = dataset
        self.classes = classes
        self.filtered_data = []

        # Filter the dataset to only include samples of the specified classes
        for image, label in self.dataset:
            if label in self.classes:
                self.filtered_data.append((image, label))

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        return self.filtered_data[idx]

# Custom Dataset for omitting specific classes
class OmitClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, omit_class=3):
        self.dataset = dataset
        self.omit_class = omit_class
        self.filtered_data = []

        # Filter the dataset to exclude samples of the omitted class
        for image, label in self.dataset:
            if label != self.omit_class:
                self.filtered_data.append((image, label))

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        return self.filtered_data[idx]

# Function to get data loader with dynamic transformations
def get_data_loader(classes=None, omit_class=None, jitter=False, batch_size=64, transform=None):
    # Default transform: to tensor
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    # If jittering is required, add random affine transformation
    if jitter:
        transform = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)), transform])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

    # Apply class filtering or omission as needed
    if classes is not None:
        train_dataset = FilteredDataset(train_dataset, classes=classes)
    elif omit_class is not None:
        train_dataset = OmitClassDataset(train_dataset, omit_class=omit_class)

    # Return the DataLoader
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
