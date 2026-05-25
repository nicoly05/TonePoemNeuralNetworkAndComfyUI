"""
train.py

Trains the ConditionalDiffusionMLP using DDPM-style denoising.

Training procedure:
  - For each latent x_0 in the dataset:
    - Sample a random timestep t
    - Sample noise ε ~ N(0, I)
    - Compute noisy latent x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    - Predict noise: ε_pred = model(x_t, t, condition)
    - Loss = MSE(ε_pred, ε)

The condition is a randomly selected OTHER latent from the dataset,
simulating the inference scenario where we condition on an incoming sound.
At inference, the condition is the actual encoded input audio.

Usage:
    python train.py
"""

import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset import LatentDataset
from model import ConditionalDiffusionMLP
from loss import DiffusionLoss

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS        = 500
BATCH_SIZE    = 32
LR            = 1e-4
T_MAX         = 1000       # diffusion timesteps
VAL_SPLIT     = 0.1
SAVE_DIR      = "models"
SAVE_PATH     = os.path.join(SAVE_DIR, "diffusion.pt")
LOG_INTERVAL  = 25
# ──────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_noise_schedule(T: int, device: torch.device):
    """
    Cosine noise schedule (better than linear for small datasets).
    Returns (alphas_cumprod,) of shape (T,).
    """
    steps = torch.arange(T + 1, dtype=torch.float32, device=device)
    f     = torch.cos(((steps / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    return alphas_cumprod[1:]   # (T,)


def add_noise(x0: torch.Tensor,
              t: torch.Tensor,
              alphas_cumprod: torch.Tensor):
    """
    Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    Returns (x_t, ε).
    """
    acp   = alphas_cumprod[t]                        # (B,)
    sqrt_acp      = acp.sqrt().view(-1, 1)
    sqrt_one_minus = (1 - acp).sqrt().view(-1, 1)
    noise = torch.randn_like(x0)
    x_t   = sqrt_acp * x0 + sqrt_one_minus * noise
    return x_t, noise


def main():
    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    ds     = LatentDataset()
    n_val  = max(1, int(len(ds) * VAL_SPLIT))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)
    print(f"Train: {n_train}  Val: {n_val}")

    # All latents for condition sampling
    all_latents = ds.latents.to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalDiffusionMLP(latent_dim=ds.latent_dim, hidden_dim=256, n_layers=4).to(device)
    criterion = DiffusionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    alphas_cumprod = make_noise_schedule(T_MAX, device)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x0   = batch.to(device)                          # (B, 128)
            B    = x0.shape[0]

            # Random timesteps
            t    = torch.randint(0, T_MAX, (B,), device=device)

            # Add noise
            x_t, noise = add_noise(x0, t, alphas_cumprod)

            # Random condition from dataset (simulates incoming audio)
            idx  = torch.randint(0, len(all_latents), (B,), device=device)
            cond = all_latents[idx]                          # (B, 128)

            # Predict noise
            noise_pred = model(x_t, t, cond)
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * B

        train_loss /= n_train
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x0   = batch.to(device)
                B    = x0.shape[0]
                t    = torch.randint(0, T_MAX, (B,), device=device)
                x_t, noise = add_noise(x0, t, alphas_cumprod)
                idx  = torch.randint(0, len(all_latents), (B,), device=device)
                cond = all_latents[idx]
                noise_pred = model(x_t, t, cond)
                val_loss  += criterion(noise_pred, noise).item() * B
        val_loss /= n_val

        if epoch % LOG_INTERVAL == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{EPOCHS}  |  "
                  f"train {train_loss:.4f}  |  val {val_loss:.4f}  |  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":           epoch,
                "state_dict":      model.state_dict(),
                "val_loss":        val_loss,
                "latent_dim":      ds.latent_dim,
                "T_max":           T_MAX,
                "alphas_cumprod":  alphas_cumprod.cpu(),
            }, SAVE_PATH)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()