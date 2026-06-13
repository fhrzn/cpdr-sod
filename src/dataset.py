import os
from typing import Literal
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random


class DUTSDataset(Dataset):
    def __init__(self, root, split: Literal["train", "val", "test"], size=256):
        """
        root:  path to DUTS-TR folder
        size:  resize target
        """
        self.size = size
        self.split = split

        dir_prefix = "TR" if split in ["train", "val"] else "TE"

        img_dir = os.path.join(root, f"DUTS-{dir_prefix}-Image")
        mask_dir = os.path.join(root, f"DUTS-{dir_prefix}-Mask")
        stems = sorted([f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg")])

        # 90/10 split for train/val monitoring
        split_idx = int(len(stems) * 0.9)
        if split == 'train':
            stems = stems[:split_idx]
        if split == "valid":
            stems = stems[split_idx:]

        self.samples = [
            (os.path.join(img_dir, s + ".jpg"), os.path.join(mask_dir, s + ".png"))
            for s in stems
        ]

        # image: resize + normalize with ImageNet stats
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # mask: resize only (no normalize)
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        if self.split in ["train", "val"]:
            # apply the same random flip to both
            if self.split == "train" and random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            img = self.img_transform(img)  # [3, H, W]
            mask = self.mask_transform(mask)  # [1, H, W], range [0,1]
            mask = (mask > 0.5).float()  # binarize

            return img, mask

        else:
            # keep original mask size for evaluation
            orig_w, orig_h = mask.size

            img = self.img_transform(img)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()  # [1, H_orig, W_orig]

            return img, mask, orig_h, orig_w
