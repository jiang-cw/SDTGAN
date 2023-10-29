import torch
import torch.nn as nn
import torch.nn.functional as F
from SwinTransformer import *

# Simplified Swin Transformer block (encoder)
class Swin_Encoder(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple(int)): Window size. Default: (7,7,7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrained=None,
                 pretrained2d=True,
                 img_size=(128, 128, 128),
                 patch_size=(4, 4, 4),
                 in_chans=4,
                 num_classes=3,
                 embed_dim=96,
                 depths=[2, 2, 2, 1],
                 depths_decoder=[1, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 frozen_stages=-1,
                 final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys3D expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{};embed_dims:{};window:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes, embed_dim, window_size))

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(img_size=img_size,
                                        patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                depths=depths,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer)),
                                      bias=False) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[2] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[2] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(
                img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]),
                dim_scale=4, dim=embed_dim)
            self.output = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self._freeze_stages()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x_downsample = []
        v_values_1 = []
        k_values_1 = []
        q_values_1 = []
        v_values_2 = []
        k_values_2 = []
        q_values_2 = []

        for i, layer in enumerate(self.layers):
            x_downsample.append(x)
            x, v1, k1, q1, v2, k2, q2 = layer(x, i)
            v_values_1.append(v1)
            k_values_1.append(k1)
            q_values_1.append(q1)
            v_values_2.append(v2)
            k_values_2.append(k2)
            q_values_2.append(q2)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2,
                            q_values_2):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], 1)
                B, C, D, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                _, _, C = x.shape
                x = x.view(B, D, H, W, C)

                x = x.permute(0, 4, 1, 2, 3)
                x = layer_up(x, v_values_1[3 - inx], k_values_1[3 - inx], q_values_1[3 - inx], v_values_2[3 - inx],
                             k_values_2[3 - inx], q_values_2[3 - inx])

        x = self.norm_up(x)

        return x

    def up_x4(self, x):
        D, H, W = self.patches_resolution
        B, _, _, _, C = x.shape

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * D, 4 * H, 4 * W, -1)
            x = x.permute(0, 4, 1, 2, 3)  # B,C,D,H,W
            x = self.output(x)

        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,
                                                                                                          self.patch_size[
                                                                                                              0], 1,
                                                                                                          1) / \
                                                self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                                   L2).permute(
                        1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)

            print(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights()
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2 = self.forward_features(
            x)
        x = self.forward_up_features(x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2,
                                     q_values_2)
        x = self.up_x4(x)

        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super(Encoder, self).__init__()
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        self.swin_block = Swin_Encoder(embed_dim, num_heads)

    def forward(self, x):
        x = self.conv(x)
        # Flatten spatial dimensions for transformer input
        x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = self.swin_block(x)
        return x.transpose(1, 2).view(x.shape[0], -1, x.shape[2], x.shape[2], x.shape[2])

class Decoder(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
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

# Sample tensor to test the models
input_tensor = torch.randn(8, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)

# Instantiate the models
encoder = Encoder(1, 128, 8)  # 1 channel input, 128-dim embedding, 8 heads
decoder = Decoder(128, 1)    # 128-dim embedding, 1 channel output
discriminator = Discriminator(1)

# Forward pass
encoded = encoder(input_tensor)
decoded = decoder(encoded)
disc_output = discriminator(decoded)

print(decoded.shape)      # Expected output shape for decoder
print(disc_output.shape)  # Expected output shape for discriminator