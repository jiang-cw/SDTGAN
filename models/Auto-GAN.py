import torch
import torch.nn as nn

# Define the Encoder-Decoder Translation Network
class TranslationNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TranslationNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, features=None):
        x = self.encoder(x)
        if features is not None:
            x = x + features
        x = self.decoder(x)
        return x

# Define the Self-representation Network (Autoencoder structure)
class SelfRepresentationNet(nn.Module):
    def __init__(self, in_channels):
        super(SelfRepresentationNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the models
translation_net = TranslationNet(1, 1)  # Assuming single-channel 3D images
self_rep_net = SelfRepresentationNet(1)
discriminator = Discriminator(1)

# Sample tensor to test the models
input_tensor = torch.randn(8, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)

# Get outputs
encoded_features, decoded_img = self_rep_net(input_tensor)
trans_output = translation_net(input_tensor, encoded_features)
disc_output = discriminator(trans_output)

print(trans_output.shape)      # Expected output shape for translation net
print(decoded_img.shape)      # Expected output shape for self representation net
print(disc_output.shape)      # Expected output shape for discriminator