import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # MultiheadAttention requires input shape (seq_len, batch_size, d_model)
        x_transposed = x.transpose(0, 1)
        attn_output, _ = self.self_attn(x_transposed, x_transposed, x_transposed)
        x = x + attn_output
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x.transpose(0, 1)


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model) for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Instantiate the models
encoder = Encoder(1, 128)  # Assuming single-channel 3D images
transformer = Transformer(128, 6)
decoder = Decoder(128, 1)
discriminator = Discriminator(1)

# Sample tensor to test the models
input_tensor = torch.randn(8, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)

# Get outputs
encoded = encoder(input_tensor)
transformed = transformer(encoded.view(encoded.shape[0], encoded.shape[1], -1))  # Flatten spatial dimensions
decoded = decoder(transformed.view(encoded.shape))
disc_output = discriminator(decoded)

print(decoded.shape)      # Expected output shape for decoder
print(disc_output.shape)  # Expected output shape for discriminator