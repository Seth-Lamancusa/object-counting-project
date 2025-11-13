import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os


def get_data_loaders(batch_size=32, data_dir='./data', image_size=32):
    """
    Get data loaders for custom data directory.
    
    Expected directory structure:
        data_dir/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                    ...
                class2/
                    img1.jpg
                    ...
            test/ (or val/)
                class1/
                    img1.jpg
                    ...
                class2/
                    img1.jpg
                    ...
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Root directory containing train/ and test/ subdirectories
        image_size: Target image size (will be resized to image_size x image_size)
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_classes: Number of classes detected
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Fallback to 'val' if 'test' doesn't exist
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_dir, 'val')
    
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )
    
    num_classes = len(train_dataset.classes)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader, num_classes


class CustomDataset(Dataset):
    """
    Custom dataset class for loading your own data.
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

