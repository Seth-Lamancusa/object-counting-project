import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json

class CountDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            # We load the 'data' key from your json
            self.data_info = json.load(f)['data']
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Get path from JSON
        img_path = self.data_info[idx]['fileName']
        
        # Convert string label (e.g., "9") to integer
        label = int(self.data_info[idx]['objectCount'])
        
        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Could not find {img_path}, returning black image.")
            image = Image.new('RGB', (256, 256)) 

        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(batch_size=16, json_file='labels.json'):
    # FIXED: Resize to 256x256 to match the Model
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = CountDataset(json_file=json_file, transform=transform)
    
    # Split 80% train, 20% test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # FIXED: Only return 2 values (train_loader, test_loader)
    return train_loader, test_loader