import torch
import matplotlib
# Use Agg backend to avoid crashes if no display window is available
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

def visualize_feature_maps(model, img_path, device, layer="conv1", save_dir=None):
    t = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()])
    
    # Safety check for the image file
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. Skipping feature map viz.")
        return

    img = Image.open(img_path).convert("RGB")
    x = t(img).unsqueeze(0).to(device)
    out = {}

    def hook(m, i, o):
        out["v"] = o.detach().cpu()

    lyr = getattr(model, layer)
    h = lyr.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(x)
    h.remove()

    f = out["v"]
    c = f.shape[1]
    cols = 8
    rows = (c // cols) + 1
    plt.figure(figsize=(12, 12))
    plt.suptitle(f"{layer} feature maps")
    for i in range(c):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(f[0, i], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"feature_maps_{layer}.png"))
        plt.close()
    else:
        plt.show()


def plot_loss_curves(train_losses, val_losses, save_dir=None):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "loss_curves.png"))
        plt.close()
    else:
        plt.show()


def show_conv_filters(model, layer_name="conv1", max_filters=12, save_dir=None):
    layer = getattr(model, layer_name)
    weights = layer.weight.data.clone()

    num_filters = min(max_filters, weights.shape[0])
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    
    # Handle single filter case
    if num_filters == 1:
        axes = [axes]

    for i in range(num_filters):
        filt = weights[i].cpu().numpy()
        axes[i].imshow(filt[0], cmap='gray')
        axes[i].set_title(f"{layer_name} #{i}")
        axes[i].axis('off')

    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"filters_{layer_name}.png"))
        plt.close()
    else:
        plt.show()


def show_predictions(model, loader, device, save_dir=None):
    model.eval()
    try:
        imgs, labels = next(iter(loader))
    except StopIteration:
        print("Loader is empty.")
        return

    outputs = model(imgs.to(device)).cpu().detach().numpy().flatten()
    preds = outputs.round()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(8):
        if i >= len(imgs): break
        axes[i].imshow(imgs[i].permute(1, 2, 0)*0.5 + 0.5)
        axes[i].set_title(
            f"True: {int(labels[i].item())}\nPred: {int(preds[i])}")
        axes[i].axis("off")

    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "predictions.png"))
        plt.close()
    else:
        plt.show()