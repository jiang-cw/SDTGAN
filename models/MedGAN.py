import torch
import torch.nn as nn

# Define the CasNet Generator (Encoder-Decoder structure for progressive refinement)
class CasNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CasNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Decoder
        self.dec1 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.final = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        out = self.final(d2)
        return out


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# Define the Pretrained Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()

        # Assume this is a simple 3D CNN, but can replace with more complex structure
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)


# Instantiate the models
generator = CasNet(1, 1)  # Assuming single-channel 3D images
discriminator = Discriminator(1)
feature_extractor = FeatureExtractor(1)

# Sample tensor to test the models
input_tensor = torch.randn(8, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)

# Get outputs
gen_output = generator(input_tensor)
disc_output = discriminator(gen_output)
features = feature_extractor(gen_output)

print(gen_output.shape)      # Expected output shape for generator
print(disc_output.shape)     # Expected output shape for discriminator
print(features.shape)        # Expected output shape for feature extractor