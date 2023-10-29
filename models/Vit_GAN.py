import torch
import torch.nn as nn

# Vision Transformer Block
class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x_transposed = x.transpose(0, 1)
        attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.mlp(x)
        x = x + ff_output
        x = self.norm2(x)
        return x

# Generator with ViT structure
class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, num_heads, num_blocks):
        super(Generator, self).__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(num_blocks)])
        self.decoder = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.decoder(x)

# Discriminator with ViT structure
class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, num_heads, num_blocks):
        super(Discriminator, self).__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(num_blocks)])
        self.decoder = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.decoder(x)

# Sample tensor to test the models
input_tensor = torch.randn(8, 64, 64, 64)  # (batch_size, depth, height, width)
input_tensor = input_tensor.view(input_tensor.shape[0], -1)  # Flatten spatial dimensions

# Instantiate the models
generator = Generator(input_tensor.shape[1], input_tensor.shape[1], 128, 8, 6)
discriminator = Discriminator(input_tensor.shape[1], 1, 128, 8, 6)

# Forward pass
generated = generator(input_tensor)
disc_output = discriminator(generated)

print(generated.shape)      # Expected output shape for generator
print(disc_output.shape)  # Expected output shape for discriminator