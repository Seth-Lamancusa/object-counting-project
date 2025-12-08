import torch
from model import SimpleCNN
from data_loader import get_data_loaders
from train import train_process
from visualize import visualize_feature_maps, plot_loss_curves, show_conv_filters, show_predictions


def main():
    JSON_FILE = 'data_mapping.json'
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

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

    train_losses, val_losses = train_process(
        model, 
        train_loader, 
        test_loader, 
        NUM_EPOCHS, 
        LEARNING_RATE, 
        device
    )

    plot_loss_curves(train_losses, val_losses)

    print("\nVisualizing learned filters...")
    show_conv_filters(model, "conv1")
    show_conv_filters(model, "conv2")

    torch.save(model.state_dict(), 'counter_model.pth')
    print("\nModel saved to 'counter_model.pth'")

    example_img = "data/1_0.jpg"
    visualize_feature_maps(model, example_img, device, "conv1")

    show_predictions(model, test_loader, device)


if __name__ == '__main__':
    main()
