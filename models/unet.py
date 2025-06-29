# import torch
# import torch.nn as nn

# def get_pos_enc(t, time_emb_dim):
#     assert time_emb_dim%2==0, "Time embedding dimension is not divisible by 2"
#     freqs = 2*torch.arange(start=0, end=time_emb_dim//2, dtype=torch.float32) / time_emb_dim//2
#     arg = t[:, None].float() * freqs[None]
#     emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
#     return emb

# def get_valid_group_count(num_channels):
#     for g in reversed(range(1, num_channels + 1)):
#         if num_channels % g == 0 and g <= 8:
#             return g
#     return 1


# class TimeEmbedding(nn.Module):
#     def __init__(self, time_emb_dim):
#         super().__init__()
#         self.time_emb_dim = time_emb_dim
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, time_emb_dim*4),
#             nn.SiLU(),
#             nn.Linear(time_emb_dim*4, time_emb_dim)
#                 )
#     def forward(self, t):
#         pos_enc = get_pos_enc(t, self.time_emb_dim)
#         time_emb = self.time_mlp(pos_enc)
#         return time_emb

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim):
#         super().__init__()
#         self.norm_in = nn.GroupNorm(get_valid_group_count(in_channels), num_channels=in_channels)
#         self.activation = nn.SiLU()
#         self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.time_proj = nn.Linear(time_emb_dim, out_channels*2)
#         self.norm_out = nn.GroupNorm(get_valid_group_count(out_channels), num_channels=out_channels)
#         self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels!=out_channels else nn.Identity()
#     def forward(self, x, time_emb):
#         h = self.norm_in(x)
#         h = self.activation(h)
#         h = self.conv_in(h)
#         gamma_beta = self.time_proj(time_emb)
#         gamma, beta = gamma_beta.chunk(2, dim=1)
#         gamma = gamma[:, :, None, None]
#         beta = beta[:, :, None, None]
#         h = h*(1+gamma) + beta
#         h = self.norm_out(h)
#         h = self.activation(h)
#         h = self.conv_out(h)

#         return h + self.residual(x)

# class AttentionBlock(nn.Module):
#     def __init__(self, out_channels, num_heads=4):
#         super().__init__()
#         self.norm = nn.GroupNorm(get_valid_group_count(out_channels), out_channels)
#         self.attn = nn.MultieadAttention(out_channels, num_heads, batch_first=True)
#         self.proj = nn.Linear(out_channels, out_channels)
#     def forward(self, x):
#         h = self.norm(x)
#         B, C, H, W = x.shape
#         h = h.view(B, C, H*W).permute(0, 2, 1)

#         attn_val, _ = self.attn(h, h, h)
#         h = self.proj(attn_val)
        
#         h = h.permute(0, 2, 1).view(B, C, H, W)
#         return x+h

# class DownSample(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim, use_conv=False):
#         super().__init__()
#         self.res_block = ResBlock(in_channels, out_channels//2, time_emb_dim)
#         self.attn_block = AttentionBlock(out_channels//2)
#         if use_conv:
#             self.downsample = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=2, padding=1)
#         else:
#             nn.AvgPool2d(2)
#             nn.Conv2d(out_channels//2, out_channels, kernel_size=1)
#     def forward(self, x, time_emb):
#         h = self.res_block(x, time_emb)
#         h = self.attn_block(h)
#         h = self.downsample(h)
#         return h

# class UpSample(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim, use_conv=False):
#         super().__init__()
#         self.res_block = ResBlock(in_channels, out_channels//2, time_emb_dim)
#         self.attn_block = AttentionBlock(out_channels//2)
#         if use_conv:
#             self.upsample = nn.ConvTranspose2d(out_channels//2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
#         else:
#             nn.Upsample(scale_factor=2)
#     def forward(self, x, time_emb):
#         h = self.res_block(x, time_emb)
#         h = self.attn_block(h)
#         h = self.upsample(h)
#         return h
    
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim, use_conv=False):
#         super().__init__()
#         self.time_embedding = TimeEmbedding(time_emb_dim)
#         self.down1 = DownSample(in_channels, 60, time_emb_dim, use_conv)
#         self.down2 = DownSample(60, 120, time_emb_dim, use_conv)
#         self.down3 = DownSample(120, 240, time_emb_dim, use_conv)
#         self.down4 = DownSample(240, 480, time_emb_dim, use_conv)
#         self.middle_block = ResBlock(480, 480, time_emb_dim)
#         self.up4 = UpSample(480, 240, time_emb_dim, use_conv)
#         self.up3 = UpSample(240, 120, time_emb_dim, use_conv)
#         self.up2 = UpSample(120, 60, time_emb_dim, use_conv)
#         self.up1 = UpSample(60, out_channels, time_emb_dim, use_conv)

#     def forward(self, x, t):
#         time_emb = self.time_embedding(t)
#         h1 = self.down1(x, time_emb)
#         h2 = self.down2(h1, time_emb)
#         h3 = self.down3(h2, time_emb)
#         h4 = self.down4(h3, time_emb)
#         h_mid = self.middle_block(h4, time_emb)
#         h_up4 = self.up4(h_mid, time_emb) + h4
#         h_up3 = self.up3(h_up4, time_emb) + h3
#         h_up2 = self.up2(h_up3, time_emb) + h2
#         h_up1 = self.up1(h_up2, time_emb) + h1

#         return h_up1

# def get_unet(cfg):
#     return UNet(
#         in_channels = cfg["in_channels"],
#         out_channels = cfg["out_channels"],
#         time_emb_dim = cfg["time_emb_dim"],
#         use_conv = cfg.get("use_conv", False),
#             )

import torch
import torch.nn as nn

def get_pos_enc(t, time_emb_dim):
    assert time_emb_dim % 2 == 0
    freqs = 2 * torch.arange(start=0, end=time_emb_dim // 2, dtype=torch.float32) / (time_emb_dim // 2)
    arg = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

def get_valid_group_count(num_channels):
    for g in reversed(range(1, num_channels + 1)):
        if num_channels % g == 0 and g <= 8:
            return g
    return 1

class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

    def forward(self, t):
        pos_enc = get_pos_enc(t, self.time_emb_dim)
        return self.time_mlp(pos_enc)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm_in = nn.GroupNorm(get_valid_group_count(in_channels), in_channels)
        self.activation = nn.SiLU()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2)
        self.norm_out = nn.GroupNorm(get_valid_group_count(out_channels), out_channels)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm_in(x)
        h = self.activation(h)
        h = self.conv_in(h)
        gamma, beta = self.time_proj(time_emb).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = h * (1 + gamma) + beta
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h + self.residual(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels < num_heads or channels == 0:
            self.use_attention = False
        else:
            self.use_attention = True
            self.norm = nn.GroupNorm(get_valid_group_count(channels), channels)
            self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
            self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        if not self.use_attention:
            return x
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        attn_val, _ = self.attn(h, h, h)
        h = self.proj(attn_val).permute(0, 2, 1).view(B, C, H, W)
        return x + h

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_conv=False):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels // 2, time_emb_dim)
        self.attn_block = AttentionBlock(out_channels // 2)
        if use_conv:
            self.downsample = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
            )

    def forward(self, x, time_emb):
        h = self.res_block(x, time_emb)
        h = self.attn_block(h)
        return self.downsample(h)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_conv=False):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels // 2, time_emb_dim)
        self.attn_block = AttentionBlock(out_channels // 2)
        if use_conv:
            self.upsample = nn.ConvTranspose2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
            )

    def forward(self, x, time_emb):
        h = self.res_block(x, time_emb)
        h = self.attn_block(h)
        return self.upsample(h)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_conv=False):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.down1 = DownSample(in_channels, 64, time_emb_dim, use_conv)
        self.down2 = DownSample(64, 128, time_emb_dim, use_conv)
        self.down3 = DownSample(128, 256, time_emb_dim, use_conv)
        self.middle_block = ResBlock(256, 256, time_emb_dim)
        self.up3 = UpSample(256, 128, time_emb_dim, use_conv)
        self.up2 = UpSample(128, 64, time_emb_dim, use_conv)
        self.up1 = UpSample(64, out_channels, time_emb_dim, use_conv)

    def forward(self, x, t):
        time_emb = self.time_embedding(t)
        h1 = self.down1(x, time_emb)
        h2 = self.down2(h1, time_emb)
        h3 = self.down3(h2, time_emb)
        h_mid = self.middle_block(h3, time_emb)
        h_up3 = self.up3(h_mid, time_emb) + h3
        h_up2 = self.up2(h_up3 + h2, time_emb)
        h_up1 = self.up1(h_up2 + h1, time_emb)
        return h_up1

def get_unet(cfg):
    return UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        time_emb_dim=cfg["time_emb_dim"],
        use_conv=cfg.get("use_conv", False),
    )
