"""Image models: CNN classifier + tiny class-conditional diffusion UNet."""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------- classifier
class ImageClassifier(nn.Module):
    """Small CNN for image classification (any square input size)."""

    def __init__(self, n_classes: int, base: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 4 * 16, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ------------------------------------------------------------------ diffusion
def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.res(x)


class TinyUNet(nn.Module):
    """Tiny class-conditional UNet predicting diffusion noise (32x32 images)."""

    def __init__(self, n_classes: int, base: int = 32, emb_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(emb_dim), nn.Linear(emb_dim, emb_dim),
            nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(n_classes, emb_dim)
        self.conv_in = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base, emb_dim)
        self.pool1 = nn.Conv2d(base, base, 4, stride=2, padding=1)
        self.down2 = ResBlock(base, base * 2, emb_dim)
        self.pool2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)
        self.mid1 = ResBlock(base * 2, base * 4, emb_dim)
        self.mid2 = ResBlock(base * 4, base * 4, emb_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base * 3, base * 2, emb_dim)  # concat(up2, h1): 2*base + base
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base * 2, base, emb_dim)
        self.out = nn.Sequential(_group_norm(base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t) + self.class_emb(y)
        h0 = self.conv_in(x)
        h1 = self.pool1(self.down1(h0, emb))
        h2 = self.pool2(self.down2(h1, emb))
        m = self.mid2(self.mid1(h2, emb), emb)
        u2 = self.dec2(torch.cat([self.up2(m), h1], dim=1), emb)
        u1 = self.dec1(torch.cat([self.up1(u2), h0], dim=1), emb)
        return self.out(u1)


class Diffusion:
    """DDPM helpers: forward noising, training step and reverse sampling."""

    def __init__(self, timesteps: int = 150, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_ac = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        s1 = self.sqrt_ac[t][:, None, None, None]
        s2 = self.sqrt_1m_ac[t][:, None, None, None]
        return s1 * x0 + s2 * noise

    def training_step(self, model: nn.Module, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t, y)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, y: torch.Tensor, img_size: int = 32) -> torch.Tensor:
        """Reverse-diffusion sampling. Returns images in [-1, 1]."""
        model.eval()
        device = next(model.parameters()).device
        x = torch.randn(len(y), 3, img_size, img_size, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((len(y),), i, dtype=torch.long, device=device)
            pred_noise = model(x, t, y.to(device))
            beta = self.betas[i]
            alpha = self.alphas[i]
            ac = self.alphas_cumprod[i]
            coef = beta / torch.sqrt(1.0 - ac)
            mean = (x - coef * pred_noise) / torch.sqrt(alpha)
            x = mean if i == 0 else mean + torch.sqrt(beta) * torch.randn_like(x)
        return x.clamp(-1, 1)
