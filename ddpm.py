import torch
from schedulers.linear import linear_scheduler
from models.unet import get_unet

class cfg:
    def __init__(self, in_channels, out_channels, time_emb_dim, use_conv):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.use_conv = use_conv

class ForwardDiffusion:
    def __init__(self, timesteps):
        self.betas = linear_scheduler(timesteps)
        alpha = 1 - self.betas
        self.alpha_bar = torch.cumprod(alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def noising(self, x, noise, t):
        sqrt_alpha_bar = self.sqrt_alpha_bar.to(x.device)[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(x.device)[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise


class ReverseDiffusion:
    def __init__(self, timesteps):
        self.betas = linear_scheduler(timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def denoising(self, xt, noise_pred, t):
        device = xt.device
        alpha = self.alphas.to(device)[t].view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bar.to(device)[t].view(-1, 1, 1, 1)

        x0 = (xt - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
        x0 = torch.clamp(x0, -1., 1.)

        mean = (xt - ((1 - alpha) * noise_pred) / (1 - alpha_bar).sqrt()) / alpha.sqrt()

        if (t == 0).all():
            return mean, x0
        else:
            t_prev = torch.clamp(t - 1, 0)
            alpha_bar_prev = self.alpha_bar.to(device)[t_prev].view(-1, 1, 1, 1)
            var = (1 - alpha_bar_prev) / (1 - alpha_bar) * self.betas.to(device)[t].view(-1, 1, 1, 1)
            sigma = torch.sqrt(var)
            z = torch.randn_like(xt)
            return mean + sigma * z, x0