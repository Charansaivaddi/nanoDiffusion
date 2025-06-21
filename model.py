import torch
import torch.nn as nn

def get_pos_enc(t, time_emb_dim):
    assert time_emb_dim%2==0, "Time embedding dimension is not divisible by 2"
    freqs = 2*torch.arange(start=0, end=time_emb_dim//2, dtype=torch.float32) / time_emb_dim//2
    arg = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
    return emb

class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
                )
    def forward(self, t):
        pos_enc = get_pos_enc(t, self.time_emb_dim)
        time_emb = self.mlp(pos_enc)
        return time_emb

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm_in = nn.GroupNorm(12, in_channels)
        self.activation = nn.SiLU()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels*2)
        self.norm_out = nn.GroupNorm(12, out_channels)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels!=out_channels else nn.Identity()
    def forward(self, x, time_emb):
        h = self.norm_in(x)
        h = self.activation(h)
        h = self.conv_in(h)
        gamma_beta = self.time_proj(time_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = h*(1+gamma) + beta
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)

        return h + self.residual(x)

