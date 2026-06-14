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
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
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
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def get_weights(self, x):
        # returns [B, 1, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

    def forward(self, x):
        return x * self.get_weights(x)


class DACF(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.se = SEModule(channels)
        self.sa = SpatialAttention()

        # stride-1 conv to downsample wF_shallow → match deep spatial size
        self.conv_down = nn.Conv2d(
            channels, channels, kernel_size=1, stride=2, bias=False
        )

        # 1x1 convs (DACF replaces 3x3 with 1x1 for efficiency)
        self.conv_inner = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.conv_outer = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, deep, shallow):
        """
        deep:    [B, C, H,   W  ]  coarse, semantic  (e.g. d3 at H/32)
        shallow: [B, C, H*2, W*2]  fine, structural  (e.g. d2 at H/16)
        returns: [B, C, H*2, W*2]  refined shallow
        """

        # 1. channel weights from deep → re-weight shallow
        w_ch = self.se.get_weights(deep)  # [B, C, 1, 1]
        wF_shallow = shallow * w_ch  # [B, C, H*2, W*2]

        # 2. spatial weights from shallow → downsample → re-weight deep
        w_sp = self.sa.get_weights(shallow)  # [B, 1, H*2, W*2]
        w_sp_down = F.interpolate(
            w_sp, size=deep.shape[2:], mode="bilinear", align_corners=False
        )  # [B, 1, H, W]
        wF_deep = deep * w_sp_down  # [B, C, H, W]

        # 3. downsample wF_shallow to deep's spatial size
        wF_shallow_down = self.conv_down(wF_shallow)  # [B, C, H, W]

        # 4. concatenate and upsample back
        Ci = torch.cat([wF_shallow_down, wF_deep], dim=1)  # [B, 2C, H, W]
        Ci_up = F.interpolate(
            Ci, size=shallow.shape[2:], mode="bilinear", align_corners=False
        )  # [B, 2C, H*2, W*2]

        # 5. fuse: equation (5) — conv_outer( conv_inner(Ci_up) + wF_shallow )
        out = self.conv_outer(self.conv_inner(Ci_up) + wF_shallow)  # [B, C, H*2, W*2]

        return out


class SegHead(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x, target_h, target_w):
        x = self.conv(x)
        x = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(x)  # [B, 1, H, W]


class EfficientNetB0Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        eff = models.efficientnet_b0(weights="IMAGENET1K_V1")

        self.stage1 = nn.Sequential(*eff.features[0:3])  # → 24ch,  H/4
        self.stage2 = nn.Sequential(*eff.features[3:4])  # → 40ch,  H/8
        self.stage3 = nn.Sequential(*eff.features[4:6])  # → 112ch, H/16
        self.stage4 = nn.Sequential(*eff.features[6:8])  # → 320ch, H/32
        # features[8] discarded

    def forward(self, x):
        f0 = self.stage1(x)  # [B, 24,  H/4,  W/4 ]
        f1 = self.stage2(f0)  # [B, 40,  H/8,  W/8 ]
        f2 = self.stage3(f1)  # [B, 112, H/16, W/16]
        f3 = self.stage4(f2)  # [B, 320, H/32, W/32]
        return f0, f1, f2, f3


class UNetDecoder(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()

        # input channels = backbone channels (no concat for deepest stage)
        # subsequent stages = upsampled prev (out_ch) + skip connection
        self.dec3 = self._make_block(320, out_ch)  # F3 only
        self.dec2 = self._make_block(out_ch + 112, out_ch)  # upsample(D3) + F2
        self.dec1 = self._make_block(out_ch + 40, out_ch)  # upsample(D2) + F1
        self.dec0 = self._make_block(out_ch + 24, out_ch)  # upsample(D1) + F0

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, f0, f1, f2, f3):
        # start from deepest, work upward
        d3 = self.dec3(f3)  # [B, 64, H/32, W/32]

        up3 = F.interpolate(
            d3, size=f2.shape[2:], mode="bilinear", align_corners=False
        )  # [B, 64, H/16, W/16]
        d2 = self.dec2(torch.cat([up3, f2], dim=1))  # [B, 64, H/16, W/16]

        up2 = F.interpolate(
            d2, size=f1.shape[2:], mode="bilinear", align_corners=False
        )  # [B, 64, H/8,  W/8 ]
        d1 = self.dec1(torch.cat([up2, f1], dim=1))  # [B, 64, H/8,  W/8 ]

        up1 = F.interpolate(
            d1, size=f0.shape[2:], mode="bilinear", align_corners=False
        )  # [B, 64, H/4,  W/4 ]
        d0 = self.dec0(torch.cat([up1, f0], dim=1))  # [B, 64, H/4,  W/4 ]

        return d0, d1, d2, d3


class ADF(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.se = SEModule(channels)

        # 3×3 conv to refine the channel-weighted shallow feature
        self.conv_shallow = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # stride-2 conv to downsample refined shallow → deep spatial size
        self.conv_down = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 1×1 conv after concat: 2C → C
        self.conv_merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, deep, shallow):
        """
        deep:    [B, C, H,   W  ]
        shallow: [B, C, H*2, W*2]
        returns:
          f_shallow: [B, C, H*2, W*2]  refined shallow
          f_deep:    [B, C, H,   W  ]  aggregated deep
        """
        # channel weights from deep → broadcast over shallow's spatial dims
        w_ch = self.se.get_weights(deep)  # [B, C, 1, 1]
        f_shallow = self.conv_shallow(shallow * w_ch)  # [B, C, H*2, W*2]

        # downsample and merge with deep
        down = self.conv_down(f_shallow)  # [B, C, H, W]
        f_deep = self.conv_merge(torch.cat([deep, down], dim=1))  # [B, C, H, W]

        return f_shallow, f_deep


class AUF(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.sa = SpatialAttention()

        # 3×3 conv to refine the spatially-weighted deep feature
        self.conv_deep = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 3×3 conv before concat with shallow
        self.conv_up = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 1×1 conv after concat: 2C → C
        self.conv_merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, deep, shallow):
        """
        deep:    [B, C, H,   W  ]
        shallow: [B, C, H*2, W*2]
        returns:
          f_deep:    [B, C, H,   W  ]  refined deep
          f_shallow: [B, C, H*2, W*2]  aggregated shallow
        """
        # spatial weights from shallow → downsample to deep's size
        w_sp = self.sa.get_weights(shallow)  # [B, 1, H*2, W*2]
        w_sp_down = F.interpolate(
            w_sp, size=deep.shape[2:], mode="bilinear", align_corners=False
        )  # [B, 1, H, W]

        # re-weight and refine deep
        f_deep = self.conv_deep(deep * w_sp_down)  # [B, C, H, W]

        # upsample refined deep → concat with shallow
        up = F.interpolate(
            f_deep, size=shallow.shape[2:], mode="bilinear", align_corners=False
        )  # [B, C, H*2, W*2]
        up = self.conv_up(up)  # [B, C, H*2, W*2]
        f_shallow = self.conv_merge(torch.cat([shallow, up], dim=1))  # [B, C, H*2, W*2]

        return f_deep, f_shallow


class CPDR_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CPDR_S"

        self.backbone = MobileNetV2Backbone()
        self.decoder = FPNDecoder()

        # DACF chain: coarse → fine
        self.dacf1 = DACF(channels=64)  # (d3, d2) → r2  at H/16
        self.dacf2 = DACF(channels=64)  # (r2, d1) → r1  at H/8

        # segmentation heads (3 outputs for deep supervision)
        self.head0 = SegHead(64)  # on d0  (H/4  — finest decoder feature)
        self.head1 = SegHead(64)  # on r1  (H/8  — DACF refined)
        self.head2 = SegHead(64)  # on r2  (H/16 — DACF refined)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # 1. backbone
        f0, f1, f2, f3 = self.backbone(x)

        # 2. decoder: normalize channels to 32
        d0, d1, d2, d3 = self.decoder(f0, f1, f2, f3)

        # 3. DACF: cross-attention fusion, coarse-to-fine
        r2 = self.dacf1(deep=d3, shallow=d2)  # [B, 32, H/16, W/16]
        r1 = self.dacf2(deep=r2, shallow=d1)  # [B, 32, H/8,  W/8 ]

        # 4. segmentation heads — all upsampled to input resolution
        out0 = self.head0(d0, H, W)  # [B, 1, H, W]
        out1 = self.head1(r1, H, W)  # [B, 1, H, W]
        out2 = self.head2(r2, H, W)  # [B, 1, H, W]

        if self.training:
            return out0, out1, out2  # all 3 for deep supervision
        else:
            return out0  # finest prediction at inference


class CPDR_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CPDR_M"

        self.backbone = EfficientNetB0Backbone()
        self.decoder = UNetDecoder(out_ch=64)

        # two ADF+AUF pairs (equivalent to CPDR-S's two DACF modules)
        self.adf1 = ADF(channels=64)
        self.auf1 = AUF(channels=64)

        self.adf2 = ADF(channels=64)
        self.auf2 = AUF(channels=64)

        # segmentation heads (deep supervision)
        self.head0 = SegHead(64)  # d0 at H/4  — finest decoder feature
        self.head1 = SegHead(64)  # r1 at H/8  — ADF+AUF refined
        self.head2 = SegHead(64)  # r2 at H/16 — ADF+AUF refined

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # 1. backbone
        f0, f1, f2, f3 = self.backbone(x)

        # 2. UNet decoder: top-down with skip connections
        d0, d1, d2, d3 = self.decoder(f0, f1, f2, f3)

        # 3. ADF+AUF pair 1: process (d3, d2) → r2
        fs2, fd3 = self.adf1(deep=d3, shallow=d2)  # channel att: d3 guides d2
        _, r2 = self.auf1(deep=fd3, shallow=fs2)  # spatial att: fs2 guides fd3 → r2

        # 4. ADF+AUF pair 2: process (r2, d1) → r1
        fs1, fd2 = self.adf2(deep=r2, shallow=d1)  # channel att: r2 guides d1
        _, r1 = self.auf2(deep=fd2, shallow=fs1)  # spatial att: fs1 guides fd2 → r1

        # 5. segmentation heads
        out0 = self.head0(d0, H, W)  # [B, 1, H, W]
        out1 = self.head1(r1, H, W)  # [B, 1, H, W]
        out2 = self.head2(r2, H, W)  # [B, 1, H, W]

        if self.training:
            return out0, out1, out2
        else:
            return out0
