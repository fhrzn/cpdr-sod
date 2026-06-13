import torch
from src.model import CPDR_S


def load_model(checkpoint_path: str, device: str = "cuda"):
    model = CPDR_S()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, MAE {ckpt['mae']:.4f}")
    return model


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
