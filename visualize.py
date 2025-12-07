import torch
import matplotlib.pyplot as plt

def show_conv_filters(model, layer_name="conv1", max_filters=12):
    layer = getattr(model, layer_name)
    weights = layer.weight.data.clone()

    num_filters = min(max_filters, weights.shape[0])
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))

    for i in range(num_filters):
        filt = weights[i].cpu().numpy()
        # Using only first channel for visualization
        axes[i].imshow(filt[0], cmap='gray')
        axes[i].set_title(f"{layer_name} #{i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def show_predictions(model, loader, device):
    model.eval()
    import matplotlib.pyplot as plt
    
    imgs, labels = next(iter(loader))
    outputs = model(imgs.to(device)).cpu().detach().numpy().flatten()
    preds = outputs.round()
    
    fig, axes = plt.subplots(2, 4, figsize=(12,6))
    axes = axes.flatten()
    
    for i in range(8):
        axes[i].imshow(imgs[i].permute(1,2,0)*0.5 + 0.5)
        axes[i].set_title(f"True labels: {labels[i]}\nPredicted labels: {int(preds[i])}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

