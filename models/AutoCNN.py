import torch
import torch.nn as nn
import numpy as np
import functools
import random

class AutoCNN(nn.Module):
    def __init__(self, in_channels, num_features):
        super(AutoCNN, self).__init__()

        # Define a single CNN module
        def cnn_module(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )
        
        self.cnn1 = cnn_module(in_channels, num_features)
        self.cnn2 = cnn_module(in_channels + num_features, num_features)
        self.cnn3 = cnn_module(in_channels + num_features, num_features)

    def forward(self, x):
        out1 = self.cnn1(x)
        out2 = self.cnn2(torch.cat([x, out1], dim=1))
        out3 = self.cnn3(torch.cat([x, out2], dim=1))
        return out3

# Test code
model = AutoCNN(1, 64)  # Assuming the input is single-channel 3D data, with 64 features in the middle
input_tensor = torch.randn(8, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)
output = model(input_tensor)

print(output.shape)  # Expected output shape: (8, 64, 64, 64, 64)