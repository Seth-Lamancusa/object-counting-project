import torch
from model import SimpleCNN
from data_loader import get_data_loaders
from train import train_process, train_one_epoch, validate
from visualize import visualize_feature_maps, train_with_curves
import torch.nn as nn
import torch.optim as optim


def main():
    # --- Configuration ---
    JSON_FILE = 'data_mapping.json'
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 1. Load Data
    print("Loading data...")
    try:
        train_loader, test_loader = get_data_loaders(
            batch_size=BATCH_SIZE, json_file=JSON_FILE)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Initialize Model
    # We have 10 classes (0 to 9 objects)
    model = SimpleCNN().to(device)

    # 3. Start Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_with_curves(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        NUM_EPOCHS
    )

    # 4. Save Model
    torch.save(model.state_dict(), 'counter_model.pth')
    print("\nModel saved to 'counter_model.pth'")

    example_img = "data/1_0.jpg"

    visualize_feature_maps(model, example_img, device, "conv1")


if __name__ == '__main__':
    main()
