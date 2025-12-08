import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import SimpleCNN
from data_loader import get_data_loaders
from train import train_one_epoch, validate
from visualize import visualize_feature_maps, train_with_curves, show_conv_filters, show_predictions


def main():
    JSON_FILE = 'data_mapping.json'
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    OUTPUT_DIR = 'output'  # Directory to save images

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    print("Loading data...")
    try:
        train_loader, test_loader = get_data_loaders(
            batch_size=BATCH_SIZE, json_file=JSON_FILE)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    model = SimpleCNN().to(device)

    # training with loss curves
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train_with_curves(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        NUM_EPOCHS,
        save_dir=OUTPUT_DIR
    )

    print("\nVisualizing learned filters...")
    show_conv_filters(model, "conv1", save_dir=OUTPUT_DIR)
    show_conv_filters(model, "conv2", save_dir=OUTPUT_DIR)

    torch.save(model.state_dict(), 'counter_model.pth')
    print("\nModel saved to 'counter_model.pth'")

    # Make sure this file exists, or change it to a valid path
    example_img = "data/1_0.jpg" 
    visualize_feature_maps(model, example_img, device, "conv1", save_dir=OUTPUT_DIR)

    print("Visualizing predictions...")
    show_predictions(model, test_loader, device, save_dir=OUTPUT_DIR)
    
    print(f"\nDone. All plots saved to ./{OUTPUT_DIR}/")


if __name__ == '__main__':
    main()