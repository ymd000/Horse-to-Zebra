"""マスクSAE本体。

    u = (x - b_pre) @ W_enc + b_enc
    u_S, u_sh = u[:, :nS], u[:, nS:]
    z_S  = ReLU(u_S)   (A' のときは hard_mask=True で 0)
    z_sh = ReLU(u_sh)  (or TopK)
    x_hat = concat(z_S, z_sh) @ W_dec + b_dec
"""
from __future__ import annotations

import torch
from torch import nn


class MaskedSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_site_latent: int,
        n_shared_latent: int,
        activation: str = "relu",
        topk: int | None = None,
    ) -> None:
        super().__init__()
        self.n_site_latent = n_site_latent
        self.n_shared_latent = n_shared_latent
        d_latent = n_site_latent + n_shared_latent

        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_latent))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))
        self.W_dec = nn.Parameter(torch.empty(d_latent, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        # decoderは W_enc の転置で初期化（よくある選択肢）
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())

        self.activation = activation
        self.topk = topk

    def encode(self, x: torch.Tensor, is_a_prime: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        u = (x - self.b_pre) @ self.W_enc + self.b_enc
        u_S, u_sh = u[:, : self.n_site_latent], u[:, self.n_site_latent :]

        z_S = torch.relu(u_S)
        if is_a_prime is not None:
            z_S = z_S * (~is_a_prime).float().unsqueeze(-1)

        if self.activation == "topk" and self.topk is not None:
            vals, idx = u_sh.topk(self.topk, dim=-1)
            z_sh = torch.zeros_like(u_sh).scatter_(-1, idx, torch.relu(vals))
        else:
            z_sh = torch.relu(u_sh)

        return z_S, z_sh

    def decode(self, z_S: torch.Tensor, z_sh: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_S, z_sh], dim=-1)
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, is_a_prime: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_S, z_sh = self.encode(x, is_a_prime)
        x_hat = self.decode(z_S, z_sh)
        return x_hat, z_S, z_sh
