from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_device, load_model
from src.model import CPDR_S


def preprocess(image_path, size=256):
    img = Image.open(image_path).convert("RGB")
    original = img.copy()  # keep for visualization

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(img).unsqueeze(0)  # [1, 3, 256, 256]
    return tensor, original


def predict(model, tensor, device="cuda"):
    tensor = tensor.to(device)
    with torch.no_grad():
        pred = model(tensor)  # [1, 1, 256, 256]
    return pred.squeeze().cpu().numpy()  # [256, 256], range [0,1]


def visualize(image_path, saliency_map, threshold=0.5, save_path=None):
    original = Image.open(image_path).convert("RGB")

    # resize original to match saliency map size for clean display
    original_resized = original.resize(
        (saliency_map.shape[1], saliency_map.shape[0]), Image.BILINEAR
    )
    original_np = np.array(original_resized)

    binary_mask = (saliency_map > threshold).astype(np.float32)

    # overlay: tint salient regions in red
    overlay = original_np.copy().astype(np.float32)
    overlay[..., 0] = np.clip(overlay[..., 0] + binary_mask * 80, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] - binary_mask * 40, 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2] - binary_mask * 40, 0, 255)
    overlay = overlay.astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(saliency_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Saliency Map")
    axes[1].axis("off")

    axes[2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Binary Mask (t={threshold})")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def run_inference(
    model: CPDR_S, image_path: str, threshold=0.5, device="cuda", save_path=None
):

    model.eval()
    model = model.to(device)

    tensor, _ = preprocess(image_path)
    saliency = predict(model, tensor, device)

    print(
        f"Saliency map — min: {saliency.min():.3f}, "
        f"max: {saliency.max():.3f}, "
        f"mean: {saliency.mean():.3f}"
    )

    visualize(image_path, saliency, threshold=threshold, save_path=save_path)
    return saliency


def main(args):

    device = get_device()

    model = load_model(args.ckpt_path, device=device)

    run_inference(
        model=model,
        image_path=args.input_img,
        threshold=args.threshold,
        device=device,
        save_path=args.output_path,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--input-img")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--output-path")
    parser.add_argument("--threshold", type=float, default=0.25)

    args = parser.parse_args()
    main(args)
