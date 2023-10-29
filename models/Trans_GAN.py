import torch
import torch.nn as nn

# Basic Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
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

# Generator with Transformer structure
class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, num_heads, num_blocks):
        super(Generator, self).__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim // (2**i), num_heads) for i in range(num_blocks)])
        self.decoder = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.embedding(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:  # Reduce embed_dim and upsample feature maps
                x = self.decoder(x)
                x = x[:, :x.size(1)//2]
        return x

# Discriminator with Transformer structure
class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, num_heads, num_blocks):
        super(Discriminator, self).__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)])
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
generator = Generator(input_tensor.shape[1], input_tensor.shape[1], 128, 8, 3)
discriminator = Discriminator(input_tensor.shape[1], 1, 128, 8, 3)

# Forward pass
generated = generator(input_tensor)
disc_output = discriminator(generated)

print(generated.shape)      # Expected output shape for generator
print(disc_output.shape)  # Expected output shape for discriminator