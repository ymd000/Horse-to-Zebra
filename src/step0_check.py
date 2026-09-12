"""SAE学習前の妥当性チェック (memo Step 0)。

1. mean(Δ) = mean(emb_a' - emb_a) と 重心差 δ = mean(emb_b) - mean(emb_a) の cos類似度・ノルム比
2. emb_a' vs emb_b の識別 AUROC (linear, rbf)
3. シマウマ判定率 (ViT 使用時のみ; ImageNet class 340)
4. Δ が定数シフトで説明できる割合 R = 1 - Var(Δ - mean(Δ)) / Var(Δ)
5. 「馬らしさ」の確率と ||Δ|| の順位相関

結果は outputs/<tag>/step0.json に保存。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import load_config, run_tag


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    cfg = load_config()
    cache_dir = Path(cfg["paths"]["cache_dir"]) / cfg["encoder"]["name"]

    emb_a = np.load(cache_dir / "emb_a.npy")
    emb_ap = np.load(cache_dir / "emb_a_prime.npy")
    emb_b = np.load(cache_dir / "emb_b.npy")

    delta = emb_ap - emb_a
    mean_delta = delta.mean(axis=0)
    centroid_shift = emb_b.mean(axis=0) - emb_a.mean(axis=0)

    metrics = {
        "n_pairs": int(len(emb_a)),
        "cos(mean_delta, centroid_shift)": cos_sim(mean_delta, centroid_shift),
        "norm_ratio": float(np.linalg.norm(mean_delta) / (np.linalg.norm(centroid_shift) + 1e-12)),
        "R_constant_shift": float(
            1 - np.var(delta - mean_delta) / (np.var(delta) + 1e-12)
        ),
    }

    # TODO (2): emb_a' vs emb_b の AUROC (LogisticRegression / SVC-rbf)
    # TODO (3): ViT-B/16 の logit を再取得してシマウマ判定率
    # TODO (5): 馬logit と ||delta|| の spearman

    out_dir = Path(cfg["paths"]["output_dir"]) / run_tag(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "step0.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
