import torch
import torch.nn as nn

# Basic convolution block for 3D data
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

# Shared Module (for feature sharing between generators)
class SharedModule(nn.Module):
    def __init__(self, channels):
        super(SharedModule, self).__init__()
        self.block = conv_block_3d(channels, channels)

    def forward(self, x1, x2):
        x_combined = x1 + x2
        return self.block(x_combined)

# Generator Module
class Generator(nn.Module):
    def __init__(self, shared_module):
        super(Generator, self).__init__()
        self.shared_module = shared_module
        self.block1 = conv_block_3d(64, 128)
        self.block2 = conv_block_3d(128, 128)
        self.block3 = conv_block_3d(128, 64)
        
    def forward(self, x, other_gen_output):
        x1 = self.block1(x)
        shared_output = self.shared_module(x1, other_gen_output)
        x2 = self.block2(shared_output)
        x3 = self.block3(x2)
        return x3

# Discriminator Module
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = conv_block_3d(64, 128)
        self.block2 = conv_block_3d(128, 1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        return x2

# Instantiate TarGAN model with shared module
shared_module = SharedModule(128)
generator_VNC = Generator(shared_module)
generator_IOM = Generator(shared_module)
discriminator_VNC = Discriminator()
discriminator_IOM = Discriminator()

# Sample tensor to test the models
input_tensor_VNC = torch.randn(8, 64, 32, 32, 32)  # (batch_size, channels, depth, height, width)
input_tensor_IOM = torch.randn(8, 64, 32, 32, 32)

# Forward pass for VNC generator
intermediate_output_VNC = generator_VNC.block1(input_tensor_VNC)
intermediate_output_IOM = generator_IOM.block1(input_tensor_IOM)

output_VNC = generator_VNC(input_tensor_VNC, intermediate_output_IOM)
output_IOM = generator_IOM(input_tensor_IOM, intermediate_output_VNC)

disc_output_VNC = discriminator_VNC(output_VNC)
disc_output_IOM = discriminator_IOM(output_IOM)

print(output_VNC.shape)      # Expected output shape for generator VNC
print(output_IOM.shape)      # Expected output shape for generator IOM
print(disc_output_VNC.shape) # Expected output shape for discriminator VNC
print(disc_output_IOM.shape) # Expected output shape for discriminator IOM