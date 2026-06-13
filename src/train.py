import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DUTSDataset
from src.eval import main as eval_fn
from src.loss import CPDRLoss
from src.model import CPDR_S
from src.utils import get_device


def make_scheduler(optimizer, warmup_epochs, total_epochs, gamma):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # linear ramp up
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        # return (1.0 - progress) ** gamma  # poly decay
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: CPDR_S,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: str,
    save_dir: str,
    warmup: int,
):

    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = make_scheduler(
        optimizer, warmup_epochs=warmup, total_epochs=num_epochs, gamma=3
    )
    criterion = CPDRLoss()

    model = model.to(device)
    best_mae = float("inf")

    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, masks in tqdm(train_loader, leave=False, desc="Train"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)  # returns (out0, out1, out2)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        mae = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, leave=False, desc="Validation"):
                imgs = imgs.to(device)
                masks = masks.to(device)

                pred = model(imgs)  # single tensor at eval
                mae += (pred - masks).abs().mean().item()

        mae /= len(val_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # logging
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] "
            f"loss: {train_loss:.4f}  "
            f"MAE: {mae:.4f}  "
            f"lr: {current_lr:.6f}  "
            f"time: {elapsed:.1f}s"
        )

        # checkpoint
        if mae < best_mae:
            best_mae = mae
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mae": best_mae,
                },
                os.path.join(save_dir, "best.pth"),
            )
            print(f"  ↑ saved best checkpoint (Loss={best_mae:.4f})")

    print(f"\nTraining done. Best MAE: {best_mae:.4f}")


def main(args):
    if args.do_eval:
        assert args.test_data is not None, (
            "Need to specify --test-data when using --do-eval flag"
        )

    device = get_device()

    train_loader = DataLoader(
        DUTSDataset(root=args.train_data, split="train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
    )

    val_loader = DataLoader(
        DUTSDataset(root=args.train_data, split="val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True,
    )

    model = CPDR_S()

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.ckpt_path,
        warmup=args.warmup,
    )

    if args.do_eval:
        eval_fn(args)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--train-data")
    parser.add_argument("--test-data", required=False)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-worker", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--do-eval", action="store_true")

    args = parser.parse_args()

    main(args)
