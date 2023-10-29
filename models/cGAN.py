import torch
import torch.nn as nn
import numpy as np
import functools
import random

# Define U-Net based Generator
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        # Encoding layers
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoding layers
        self.dec1 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256*2, 128)
        self.dec3 = self.conv_block(128*2, 64)
        self.final = nn.Conv3d(64*2, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], 1))
        d3 = self.dec3(torch.cat([d2, e2], 1))
        out = self.final(torch.cat([d3, e1], 1))
        
        return out


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# Test code
generator = Generator(1, 1)  # Assuming single-channel 3D images
discriminator = Discriminator(1)

input_tensor = torch.randn(8, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)
gen_output = generator(input_tensor)
disc_output = discriminator(gen_output)

print(gen_output.shape)  # Expected output shape for generator: (8, 1, 64, 64, 64)
print(disc_output.shape)  # Expected output shape for discriminator: (8, 1, 4, 4, 4)