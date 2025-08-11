import torch
import torch.nn as nn

"""
This file contains model blocks and model itself.
Defines the Attention - UNet architecture and 
its building blocks for semantic segmentation of the Mila logos. 
"""


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """ 
        Performs two consecutive 3×3 convolutions 
        each followed by a ReLU activation.
        """
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    """ 
    Performs a DoubleConv block followed by 
    2×2 max pooling to reduce spatial resolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class AttentionBlock(nn.Module):
    """ 
    Applies an attention mechanism on the skip connection features.
    Typical attention gate as introduced in 
    'Attention U-Net: Learning Where to Look for the Pancreas' (O. Oktay et al).
    F_g: Number of channels in gating signal (from the decoder).
    F_l: Number of channels in the skip connection (from the encoder).
    F_int: Number of intermediate channels.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        """
        g: gating signal (up-sampled feature from decoder)
        x: skip connection (feature from encoder)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Combine gating and skip features
        psi = self.psi(g1 + x1)
        # Element-wise multiplication: scale skip connection by attention
        return x * psi


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A transposed convolution to increase spatial resolution, then concatenates with the 
        corresponding encoder feature map for reconstruction.
        in_channels: # of channels for the incoming feature (decoder)
        out_channels: # of channels for the skip connection and post-concat conv
        """
        super().__init__()
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Attention Gate (AG):
        #  - Gating signal has in_channels//2 (the up-sampled feature).
        #  - Skip connection has out_channels channels from the encoder.
        #  - We pick F_int in a straightforward way (e.g., out_channels//2).
        self.attention = AttentionBlock(F_g=in_channels // 2,
                                        F_l=out_channels,
                                        F_int=out_channels // 2)

        # Final convolution after concatenation
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: decoder feature map (gating signal)
        x2: encoder feature map (skip connection)
        """
        x1 = self.up(x1)          # up-sampling
        x2 = self.attention(x1, x2)  # apply attention gate on the skip
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Keeps the UNet components and dimensions from 'U-Net: Convolutional Networks for Biomedical Image Segmentation' by
    Ronneberger et al. and adds the attnetion to the skip connection inspired by 
    'Attention U-Net: Learning Where to Look for the Pancreas' by Oktay et al.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        # Bottleneck
        b = self.bottle_neck(p4)

        # Decoder with attention in skip connections
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
