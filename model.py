import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for image classification.
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # 256x256 image size -> 16x16 after 4 pools
        self.fc2 = nn.Linear(512, num_classes)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional block 1
        x = self.pool(self.relu(self.conv1(x)))
        
        # Convolutional block 2
        x = self.pool(self.relu(self.conv2(x)))
        
        # Convolutional block 3
        x = self.pool(self.relu(self.conv3(x)))

        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

