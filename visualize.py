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
