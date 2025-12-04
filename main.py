import torch
from model import SimpleCNN
from data_loader import get_data_loaders
from train import train_process

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
        train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE, json_file=JSON_FILE)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Initialize Model
    # We have 10 classes (0 to 9 objects)
    model = SimpleCNN().to(device)

    # 3. Start Training
    train_process(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # 4. Save Model
    torch.save(model.state_dict(), 'counter_model.pth')
    print("\nModel saved to 'counter_model.pth'")

if __name__ == '__main__':
    main()