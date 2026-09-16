"""
UNet Segmentation Model
=======================
Standard UNet with optional bilinear upsampling.
Extracted from train_unet.py so different models can be swapped in
by creating additional files in this folder and importing them in train_unet.py.

To add a new model:
    1. Create models/your_model.py with a class that follows the same
       interface: __init__(n_channels, n_classes) + forward(x) -> logits
    2. In train_unet.py change the import at the top:
           from models.your_model import YourModel as SegModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  BUILDING BLOCKS
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) × 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,  mid_channels,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1    = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1    = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────
#  UNET
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Classic UNet.

    Args:
        n_channels (int): Number of input image channels (e.g. 3 for RGB).
        n_classes  (int): Number of output segmentation classes (including background).
        bilinear   (bool): Use bilinear upsampling instead of transposed convolutions.
    """

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512  // factor, bilinear)
        self.up2   = Up(512,  256  // factor, bilinear)
        self.up3   = Up(256,  128  // factor, bilinear)
        self.up4   = Up(128,  64,             bilinear)
        self.outc          = OutConv(64, n_classes)
        self.checkpointing = False

    def forward(self, x):
        if self.checkpointing:
            from torch.utils.checkpoint import checkpoint
            ck = lambda fn, *a: checkpoint(fn, *a, use_reentrant=False)
            x1 = ck(self.inc,   x)
            x2 = ck(self.down1, x1)
            x3 = ck(self.down2, x2)
            x4 = ck(self.down3, x3)
            x5 = ck(self.down4, x4)
            x  = ck(self.up1,   x5, x4)
            x  = ck(self.up2,   x,  x3)
            x  = ck(self.up3,   x,  x2)
            x  = ck(self.up4,   x,  x1)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x  = self.up1(x5, x4)
            x  = self.up2(x,  x3)
            x  = self.up3(x,  x2)
            x  = self.up4(x,  x1)
        return self.outc(x)

    def use_checkpointing(self):
        """Enable gradient checkpointing (saves GPU memory, slightly slower)."""
        self.checkpointing = True