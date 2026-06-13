import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
    
        mv2 = models.mobilenet_v2(weights="IMAGENET1K_V1")

        self.stage1 = nn.Sequential(*mv2.features[0:4])
        self.stage2 = nn.Sequential(*mv2.features[4:7])
        self.stage3 = nn.Sequential(*mv2.features[7:14])
        self.stage4 = nn.Sequential(*mv2.features[14:17])

    def forward(self, x):
        f0 = self.stage1(x)
        f1 = self.stage2(f0)
        f2 = self.stage3(f1)
        f3 = self.stage4(f2)

        return f0, f1, f2, f3
    
class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec0 = self._conv(24, 64)
        self.dec1 = self._conv(32, 64)
        self.dec2 = self._conv(96, 64)
        self.dec3 = self._conv(160, 64)

    def _conv(self, in_ch, out_ch):
        return nn.Sequential(
            # conv 1
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # conv 2
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # conv 3
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, f0, f1, f2, f3):
        d0 = self.dec0(f0)
        d1 = self.dec1(f1)
        d2 = self.dec2(f2)
        d3 = self.dec3(f3)

        return d0, d1, d2, d3

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def get_weights(self, x):
        # returns [B, C, 1, 1] — broadcasts to any spatial size
        B, C, _, _ = x.shape
        w = self.excite(self.squeeze(x))
        return w.view(B, C, 1, 1)

    def forward(self, x):
        return x * self.get_weights(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def get_weights(self, x):
        # returns [B, 1, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _  = torch.max(x,  dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

    def forward(self, x):
        return x * self.get_weights(x)

class DACF(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.se = SEModule(channels)
        self.sa = SpatialAttention()

        # stride-1 conv to downsample wF_shallow → match deep spatial size
        self.conv_down  = nn.Conv2d(channels, channels, kernel_size=1, stride=2, bias=False)

        # 1x1 convs (DACF replaces 3x3 with 1x1 for efficiency)
        self.conv_inner = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.conv_outer = nn.Conv2d(channels,     channels, kernel_size=1, bias=False)

    def forward(self, deep, shallow):
        """
        deep:    [B, C, H,   W  ]  coarse, semantic  (e.g. d3 at H/32)
        shallow: [B, C, H*2, W*2]  fine, structural  (e.g. d2 at H/16)
        returns: [B, C, H*2, W*2]  refined shallow
        """

        # 1. channel weights from deep → re-weight shallow
        w_ch       = self.se.get_weights(deep)                      # [B, C, 1, 1]
        wF_shallow = shallow * w_ch                                  # [B, C, H*2, W*2]

        # 2. spatial weights from shallow → downsample → re-weight deep
        w_sp       = self.sa.get_weights(shallow)                    # [B, 1, H*2, W*2]
        w_sp_down  = F.interpolate(w_sp, size=deep.shape[2:],
                                   mode='bilinear', align_corners=False)  # [B, 1, H, W]
        wF_deep    = deep * w_sp_down                                # [B, C, H, W]

        # 3. downsample wF_shallow to deep's spatial size
        wF_shallow_down = self.conv_down(wF_shallow)                 # [B, C, H, W]

        # 4. concatenate and upsample back
        Ci    = torch.cat([wF_shallow_down, wF_deep], dim=1)         # [B, 2C, H, W]
        Ci_up = F.interpolate(Ci, size=shallow.shape[2:],
                              mode='bilinear', align_corners=False)   # [B, 2C, H*2, W*2]

        # 5. fuse: equation (5) — conv_outer( conv_inner(Ci_up) + wF_shallow )
        out = self.conv_outer(self.conv_inner(Ci_up) + wF_shallow)   # [B, C, H*2, W*2]

        return out

class SegHead(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x, target_h, target_w):
        x = self.conv(x)
        x = F.interpolate(x, size=(target_h, target_w),
                          mode='bilinear', align_corners=False)
        return torch.sigmoid(x)   # [B, 1, H, W]

class CPDR_S(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = MobileNetV2Backbone()
        self.decoder  = FPNDecoder()

        # DACF chain: coarse → fine
        self.dacf1 = DACF(channels=64)   # (d3, d2) → r2  at H/16
        self.dacf2 = DACF(channels=64)   # (r2, d1) → r1  at H/8

        # segmentation heads (3 outputs for deep supervision)
        self.head0 = SegHead(64)   # on d0  (H/4  — finest decoder feature)
        self.head1 = SegHead(64)   # on r1  (H/8  — DACF refined)
        self.head2 = SegHead(64)   # on r2  (H/16 — DACF refined)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # 1. backbone
        f0, f1, f2, f3 = self.backbone(x)

        # 2. decoder: normalize channels to 32
        d0, d1, d2, d3 = self.decoder(f0, f1, f2, f3)

        # 3. DACF: cross-attention fusion, coarse-to-fine
        r2 = self.dacf1(deep=d3, shallow=d2)   # [B, 32, H/16, W/16]
        r1 = self.dacf2(deep=r2, shallow=d1)   # [B, 32, H/8,  W/8 ]

        # 4. segmentation heads — all upsampled to input resolution
        out0 = self.head0(d0, H, W)   # [B, 1, H, W]
        out1 = self.head1(r1, H, W)   # [B, 1, H, W]
        out2 = self.head2(r2, H, W)   # [B, 1, H, W]

        if self.training:
            return out0, out1, out2   # all 3 for deep supervision
        else:
            return out0               # finest prediction at inference