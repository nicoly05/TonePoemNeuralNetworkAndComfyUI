"""
model.py

Contains both model architectures used by this project:

  LatentRefiner — used by inference.py (retrieval version)
    Small denoising MLP that cleans up blended latent vectors.
    Input/output: (batch, 128). No conditioning.

  ConditionalDiffusionMLP — used by inference_generative.py
    DDPM noise prediction network conditioned on an input latent.
    Generates novel latents that have never existed in the dataset.
"""

import torch
import torch.nn as nn
import math

LATENT_DIM = 128


# ── Retrieval model ───────────────────────────────────────────────────────────

class LatentRefiner(nn.Module):
    """
    Small residual MLP used by the retrieval inference engine.
    Takes a blended/noisy latent vector, returns a cleaned-up version.
    Input and output: (batch, latent_dim).
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.net(x)


# ── Generative model ──────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Encodes the diffusion timestep as a sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.cond_scale = nn.Linear(dim, dim)
        self.cond_shift = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.cond_scale(cond)
        shift = self.cond_shift(cond)
        return x + self.net(x * (1 + scale) + shift)


class ConditionalDiffusionMLP(nn.Module):
    """
    DDPM noise prediction network used by the generative inference engine.

    Inputs:
        x_t      : noisy latent at timestep t  (B, latent_dim)
        t        : diffusion timestep           (B,)
        condition : encoded input audio latent  (B, latent_dim)
    Output:
        predicted noise                         (B, latent_dim)
    """

    def __init__(self,
                 latent_dim: int = LATENT_DIM,
                 hidden_dim: int = 512,
                 time_dim:   int = 128,
                 n_layers:   int = 6):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        self.cond_proj  = nn.Linear(latent_dim, hidden_dim)
        self.input_proj = nn.Linear(latent_dim + time_dim, hidden_dim)
        self.layers     = nn.ModuleList([ResidualBlock(hidden_dim)
                                         for _ in range(n_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_t: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h     = self.input_proj(torch.cat([x_t, t_emb], dim=-1))
        cond  = self.cond_proj(condition)
        h     = h + cond
        for layer in self.layers:
            h = layer(h, cond)
        return self.out(h)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── LatentRefiner ──")
    refiner = LatentRefiner()
    x = torch.randn(8, LATENT_DIM)
    y = refiner(x)
    print(f"  Input:  {x.shape}  Output: {y.shape}")
    print(f"  Params: {sum(p.numel() for p in refiner.parameters()):,}")

    print("\n── ConditionalDiffusionMLP ──")
    diffusion = ConditionalDiffusionMLP()
    x_t   = torch.randn(8, LATENT_DIM)
    t     = torch.randint(0, 1000, (8,))
    cond  = torch.randn(8, LATENT_DIM)
    out   = diffusion(x_t, t, cond)
    print(f"  Input:  {x_t.shape}  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in diffusion.parameters()):,}")