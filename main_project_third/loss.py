"""
loss.py

Diffusion training loss — simple MSE on predicted vs actual noise.

We use the standard DDPM objective: at each training step, sample a
random timestep t, add that much noise to a clean latent, ask the model
to predict the noise, and minimise MSE between prediction and truth.
"""

import torch
import torch.nn as nn


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, noise_pred: torch.Tensor,
                noise_true: torch.Tensor) -> torch.Tensor:
        return self.mse(noise_pred, noise_true)


if __name__ == "__main__":
    criterion = DiffusionLoss()
    pred  = torch.randn(8, 128)
    truth = torch.randn(8, 128)
    print(f"Loss (random): {criterion(pred, truth):.4f}")
    print(f"Loss (perfect): {criterion(truth, truth):.4f}")