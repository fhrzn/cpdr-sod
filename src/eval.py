import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import CPDR_M, CPDR_S
from src.dataset import DUTSDataset
from src.utils import get_device, load_model


def evaluate(model: CPDR_S | CPDR_M, test_loader: DataLoader, device: str):
    model.eval()
    model = model.to(device)

    mae_total = 0.0
    num_thresholds = 256
    prec_sum = np.zeros(num_thresholds)
    rec_sum = np.zeros(num_thresholds)
    n = 0

    with torch.no_grad():
        for imgs, masks, orig_h, orig_w in tqdm(test_loader, desc="evaluate"):
            imgs = imgs.to(device)

            pred = model(imgs)  # [1, 1, 256, 256]

            # upsample prediction to original mask size
            orig_h, orig_w = orig_h.item(), orig_w.item()
            pred = F.interpolate(
                pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )

            pred_np = pred.squeeze().cpu().numpy()  # [H, W]
            gt_np = masks.squeeze().numpy()  # [H, W]

            # ── MAE ──────────────────────────────────────────────
            mae_total += np.abs(pred_np - gt_np).mean()

            # ── F-measure across thresholds ───────────────────────
            gt_bin = (gt_np > 0.5).astype(np.float32)
            for i, thresh in enumerate(np.linspace(0, 1, num_thresholds)):
                pred_bin = (pred_np >= thresh).astype(np.float32)
                tp = (pred_bin * gt_bin).sum()
                fp = (pred_bin * (1 - gt_bin)).sum()
                fn = ((1 - pred_bin) * gt_bin).sum()
                prec_sum[i] += tp / (tp + fp + 1e-8)
                rec_sum[i] += tp / (tp + fn + 1e-8)

            n += 1

    # aggregate metrics
    mae = mae_total / n

    beta_sq = 0.3
    prec = prec_sum / n
    rec = rec_sum / n
    fm = (1 + beta_sq) * prec * rec / (beta_sq * prec + rec + 1e-8)
    mean_fm = fm.mean()
    max_fm = fm.max()

    print(f"\n── DUTS-TE Results ({n} images) ──────────")
    print(f"  MAE:         {mae:.4f}")
    print(f"  Mean Fβ:     {mean_fm:.4f}")
    print(f"  Max  Fβ:     {max_fm:.4f}")

    return {"mae": mae, "mean_fm": mean_fm, "max_fm": max_fm}


def main(args):
    device = get_device()

    test_loader = DataLoader(
        DUTSDataset(root=args.test_data, split="test"),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_worker,
    )

    model = load_model(args.ckpt_path, device=device)

    evaluate(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--test-data")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--num-worker", type=int, default=4)

    args = parser.parse_args()

    main(args)
