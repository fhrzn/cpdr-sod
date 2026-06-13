import torch.nn as nn


class DICELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        total = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

        dice = (2 * intersection + self.eps) / (total + self.eps)  # ← added 2*
        return 1 - dice.mean()


class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        return 1 - iou.mean()


class CPDRLoss(nn.Module):
    """Combined DICE + IoU with deep supervision across all 3 heads."""

    def __init__(self):
        super().__init__()
        self.dice = DICELoss()
        self.iou = IoULoss()

    def _single(self, pred, target):
        return self.dice(pred, target) + self.iou(pred, target)

    def forward(self, preds, target):
        """
        preds:  tuple of (out0, out1, out2) during training
                or single tensor at inference
        target: [B, 1, H, W] binary ground truth mask
        """
        out0, out1, out2 = preds
        loss = (
            self._single(out0, target)
            + self._single(out1, target)
            + self._single(out2, target)
        )
        return loss
