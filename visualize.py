import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from train import train_one_epoch, validate


def visualize_feature_maps(model, img_path, device, layer="conv1"):
    t = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()])
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
    plt.show()


def train_with_curves(model, train_loader, test_loader, optimizer, criterion, device, epochs):
    tr = []
    vl = []
    for e in range(epochs):
        a, _ = train_one_epoch(model, train_loader,
                               criterion, optimizer, device)
        b, _ = validate(model, test_loader, criterion, device)
        tr.append(a)
        vl.append(b)
    plt.figure(figsize=(7, 5))
    plt.plot(tr, label="train")
    plt.plot(vl, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train vs val loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
