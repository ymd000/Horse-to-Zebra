"""SAE を A と A' のペアで学習。

損失:
    mse(x_hat_a, x_a) + mse(x_hat_a', x_a')
    + lam_l1 * (|z_S_a|_1 + |z_sh_a|_1 + |z_sh_a'|_1)
    + lam_pair * ||z_sh_a - z_sh_a'||^2

保存:
    <output_dir>/<tag>/ckpt.pt
    <output_dir>/<tag>/config_snapshot.yaml  (再現用に完全な cfg を書き出し)
    <output_dir>/<tag>/train_log.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from config import load_config, run_tag
from sae_model import MaskedSAE


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg["train"]["seed"])

    cache_dir = Path(cfg["paths"]["cache_dir"]) / cfg["encoder"]["name"]
    emb_a = torch.from_numpy(np.load(cache_dir / "emb_a.npy")).float()
    emb_ap = torch.from_numpy(np.load(cache_dir / "emb_a_prime.npy")).float()
    d_in = emb_a.shape[1]

    d_latent = int(cfg["sae"]["expansion"] * d_in)
    n_site = cfg["sae"]["n_site_latent"]
    model = MaskedSAE(
        d_in=d_in,
        n_site_latent=n_site,
        n_shared_latent=d_latent - n_site,
        activation=cfg["sae"]["activation"],
        topk=cfg["sae"]["topk"],
    ).to(cfg["encoder"]["device"])

    loader = DataLoader(
        TensorDataset(emb_a, emb_ap),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    lam_l1 = cfg["train"]["lam_l1"]
    lam_pair = cfg["train"]["lam_pair"]
    hard_mask = cfg["train"]["hard_mask"]
    device = cfg["encoder"]["device"]

    log = []
    for epoch in range(cfg["train"]["epochs"]):
        total = 0.0
        for x_a, x_ap in loader:
            x_a, x_ap = x_a.to(device), x_ap.to(device)
            is_ap_false = torch.zeros(x_a.size(0), dtype=torch.bool, device=device)
            is_ap_true = torch.ones(x_ap.size(0), dtype=torch.bool, device=device)

            xa_hat, zS_a, zsh_a = model(x_a, is_ap_false if hard_mask else None)
            xap_hat, zS_ap, zsh_ap = model(x_ap, is_ap_true if hard_mask else None)

            recon = torch.nn.functional.mse_loss(xa_hat, x_a) + torch.nn.functional.mse_loss(xap_hat, x_ap)
            l1 = zS_a.abs().sum(-1).mean() + zsh_a.abs().sum(-1).mean() + zsh_ap.abs().sum(-1).mean()
            pair = ((zsh_a - zsh_ap) ** 2).sum(-1).mean()
            loss = recon + lam_l1 * l1 + lam_pair * pair

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x_a.size(0)

        log.append({"epoch": epoch, "loss": total / len(loader.dataset)})
        print(f"epoch {epoch}: loss={log[-1]['loss']:.4f}")

    out_dir = Path(cfg["paths"]["output_dir"]) / run_tag(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "ckpt.pt")
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    import pandas as pd
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
