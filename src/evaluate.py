"""学習済み SAE の定量評価。

指標:
    - Δ 説明率: 1 - ||delta - contrib_S||^2 / ||delta||^2
        contrib_S = (z_S_a' - z_S_a) @ W_dec_S  (ハードマスク時は -z_S_a @ W_dec_S)
    - 施設識別 AUROC (z_S のみ): 本物 A/B を分離できるか
    - 施設識別 AUROC (z_sh のみ): 内容側への漏れ確認 (0.5 に近いほど良い)
    - ペア検索: emb_shared_a に最も近い emb_a' が自分のペアかどうか (recall@1)

出力:
    <output_dir>/<tag>/eval.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from config import load_config, run_tag
from sae_model import MaskedSAE


def main() -> None:
    cfg = load_config()
    cache_dir = Path(cfg["paths"]["cache_dir"]) / cfg["encoder"]["name"]
    out_dir = Path(cfg["paths"]["output_dir"]) / run_tag(cfg)

    emb_a = torch.from_numpy(np.load(cache_dir / "emb_a.npy")).float()
    emb_ap = torch.from_numpy(np.load(cache_dir / "emb_a_prime.npy")).float()
    emb_b = torch.from_numpy(np.load(cache_dir / "emb_b.npy")).float()
    d_in = emb_a.shape[1]

    d_latent = int(cfg["sae"]["expansion"] * d_in)
    n_site = cfg["sae"]["n_site_latent"]
    model = MaskedSAE(
        d_in=d_in,
        n_site_latent=n_site,
        n_shared_latent=d_latent - n_site,
        activation=cfg["sae"]["activation"],
        topk=cfg["sae"]["topk"],
    )
    model.load_state_dict(torch.load(out_dir / "ckpt.pt", map_location="cpu"))
    model.eval()

    metrics: dict = {}

    # TODO: Δ 説明率
    # TODO: 施設識別 AUROC (z_S / z_sh)
    # TODO: ペア検索 recall@1
    # TODO: 比較対象 (重心補正 / Ridge / A' 教師) を並べる

    with open(out_dir / "eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
