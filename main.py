"""
Main script to run CNN training.
"""
import torch
from model import SimpleCNN
from data_loader import get_data_loaders
from train import train_model


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    data_dir = './data'  # Change this to your data directory
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    image_size = 32  # Input image size
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader, num_classes = get_data_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
        image_size=image_size
    )
    print(f"Found {num_classes} classes")
    
    # Initialize model
    print("\nInitializing model...")
    model = SimpleCNN(num_classes=num_classes, input_channels=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print("\nModel saved to 'model.pth'")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

