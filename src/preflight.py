"""horse2zebra マスクSAE実験の事前確認。

確認1: 変化方向の一致 (cos類似度・ノルム比・サンプルごとcosヒストグラム)
確認2: A' vs B 識別AUROC (linear / rbf, train→test, 5 seed)
確認3: シマウマ判定率 (ViT-B/16 使用時のみ)
確認4: 定数シフトで説明できる割合 R・方向揃い度・長さばらつき
確認5: Δ大小と画像内容の関係 (可視化グリッド、確認4次第)

保存先:
    <output_dir>/preflight/<encoder>/preflight.json
    <output_dir>/preflight/<encoder>/check1_hist.png
    <output_dir>/preflight/<encoder>/check5_grid.png  (条件付き)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Liberation Sans", "FreeSans", "Arial", "sans-serif"
]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import load_config


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ─── 確認1 ────────────────────────────────────────────────────────────────────

def _check1(
    delta: np.ndarray,
    mean_delta: np.ndarray,
    delta_real: np.ndarray,
    out_dir: Path,
) -> dict:
    norms = np.linalg.norm(delta, axis=1)
    per_cos = (delta @ mean_delta) / (norms * np.linalg.norm(mean_delta) + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_cos, bins=40, edgecolor="white", linewidth=0.5)
    ax.axvline(per_cos.mean(), color="r", linestyle="--", label=f"mean={per_cos.mean():.3f}")
    ax.set_xlabel("cos(delta_i, mean_delta)")
    ax.set_ylabel("count")
    ax.set_title("Check1: per-sample cosine similarity to mean_delta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "check1_hist.png", dpi=150)
    plt.close(fig)

    return {
        "cos_mean_delta_vs_delta_real": _cos_sim(mean_delta, delta_real),
        "norm_ratio": float(np.linalg.norm(mean_delta) / (np.linalg.norm(delta_real) + 1e-12)),
        "per_sample_cos_mean": float(per_cos.mean()),
        "per_sample_cos_std": float(per_cos.std()),
    }


# ─── 確認2 ────────────────────────────────────────────────────────────────────

def _auroc_pair(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    kernels: list[str],
    seeds: list[int],
) -> dict:
    out: dict = {}
    for kernel in kernels:
        aucs: list[float] = []
        for seed in seeds:
            if kernel == "linear":
                clf: Pipeline = Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)),
                ])
                clf.fit(X_tr, y_tr)
                scores = clf.decision_function(X_te)
            else:
                base: Pipeline = Pipeline([
                    ("sc", StandardScaler()),
                    ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
                ])
                gs = GridSearchCV(
                    base,
                    {"svc__gamma": ["scale", "auto", 0.01, 0.1, 1.0]},
                    cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                gs.fit(X_tr, y_tr)
                scores = gs.best_estimator_.decision_function(X_te)
            aucs.append(float(roc_auc_score(y_te, scores)))
        out[kernel] = {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "runs": aucs}
    return out


def _check2(
    emb_a: np.ndarray, emb_ap: np.ndarray, emb_b: np.ndarray,
    emb_a_te: np.ndarray, emb_ap_te: np.ndarray, emb_b_te: np.ndarray,
    kernels: list[str],
    seeds: list[int],
) -> dict:
    pairs = [
        ("A_vs_B",  emb_a,  emb_b,  emb_a_te,  emb_b_te),
        ("A_vs_Ap", emb_a,  emb_ap, emb_a_te,  emb_ap_te),
        ("Ap_vs_B", emb_ap, emb_b,  emb_ap_te, emb_b_te),
    ]
    out: dict = {}
    for name, X0_tr, X1_tr, X0_te, X1_te in pairs:
        X_tr = np.concatenate([X0_tr, X1_tr])
        y_tr = np.array([0] * len(X0_tr) + [1] * len(X1_tr))
        X_te = np.concatenate([X0_te, X1_te])
        y_te = np.array([0] * len(X0_te) + [1] * len(X1_te))
        out[name] = _auroc_pair(X_tr, y_tr, X_te, y_te, kernels, seeds)
    return out


# ─── 確認3 ────────────────────────────────────────────────────────────────────

def _zebra_rates(paths: list[Path], device: str, batch_size: int = 64) -> dict:
    import timm
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, paths: list[Path], transform: object) -> None:
            self.paths, self.transform = paths, transform
        def __len__(self) -> int: return len(self.paths)
        def __getitem__(self, i: int) -> object:
            return self.transform(Image.open(self.paths[i]).convert("RGB"))  # type: ignore[operator]

    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.to(device).eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    loader = DataLoader(_DS(paths, transform), batch_size=batch_size, num_workers=4)
    probs_zebra: list[float] = []
    top1_zebra: list[bool] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probs = torch.softmax(logits, dim=-1)
            probs_zebra.extend(probs[:, 340].cpu().tolist())
            top1_zebra.extend((logits.argmax(dim=-1) == 340).cpu().tolist())

    arr = np.array(probs_zebra)
    return {
        "top1_zebra_rate": float(np.mean(top1_zebra)),
        "mean_prob": float(arr.mean()),
        "median_prob": float(np.median(arr)),
    }


def _check3(h2z: Path, filenames_a_te: list[str], device: str) -> dict:
    paths_ap_te = [h2z / "testA_prime" / n for n in filenames_a_te]
    paths_b_te  = sorted((h2z / "testB").glob("*.jpg"))
    paths_a_te  = [h2z / "testA" / n for n in filenames_a_te]
    return {
        "A_test":       _zebra_rates(paths_a_te,  device),
        "A_prime_test": _zebra_rates(paths_ap_te, device),
        "B_test":       _zebra_rates(paths_b_te,  device),
    }


# ─── 確認4 ────────────────────────────────────────────────────────────────────

def _check4(delta: np.ndarray, mean_delta: np.ndarray) -> dict:
    residual = delta - mean_delta
    R = float(1 - (residual ** 2).sum() / ((delta ** 2).sum() + 1e-12))
    norms = np.linalg.norm(delta, axis=1)
    direction_alignment = float(
        (delta @ mean_delta / (norms * np.linalg.norm(mean_delta) + 1e-12)).mean()
    )
    length_variation_cv = float(norms.std() / (norms.mean() + 1e-12))
    return {
        "R_constant_shift": R,
        "direction_alignment": direction_alignment,
        "length_variation_cv": length_variation_cv,
    }


# ─── 確認5 ────────────────────────────────────────────────────────────────────

def _check5(
    delta: np.ndarray,
    filenames: list[str],
    src_dir: Path,
    out_dir: Path,
) -> None:
    norms = np.linalg.norm(delta, axis=1)
    order = norms.argsort()
    bottom10 = order[:10]
    top10    = order[-10:][::-1]

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for col, (ti, bi) in enumerate(zip(top10, bottom10)):
        for row, idx in enumerate([ti, bi]):
            img = Image.open(src_dir / filenames[idx]).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{norms[idx]:.2f}", fontsize=7)

    fig.text(0.01, 0.75, "large ||delta||", va="center", fontsize=9, color="C1")
    fig.text(0.01, 0.25, "small ||delta||", va="center", fontsize=9, color="C0")
    fig.suptitle("Check5: image content vs ||delta|| (top: large / bottom: small)", fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 1])
    fig.savefig(out_dir / "check5_grid.png", dpi=120)
    plt.close(fig)


# ─── UMAP ────────────────────────────────────────────────────────────────────

def _umap_plot(
    emb_a: np.ndarray,
    emb_ap: np.ndarray,
    emb_b: np.ndarray,
    out_dir: Path,
    seed: int = 0,
) -> None:
    import umap

    # A / A' / B 全部まとめて fit し、一貫した座標系を得る
    na, nap, nb = len(emb_a), len(emb_ap), len(emb_b)
    all_emb = np.concatenate([emb_a, emb_ap, emb_b])
    reducer = umap.UMAP(n_components=2, random_state=seed)
    xy = reducer.fit_transform(all_emb)
    xy_a, xy_ap, xy_b = xy[:na], xy[na:na + nap], xy[na + nap:]

    kw = dict(s=8, alpha=0.6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: A vs B
    axes[0].scatter(xy_a[:, 0],  xy_a[:, 1],  label="A (horse)",   color="C0", **kw)
    axes[0].scatter(xy_b[:, 0],  xy_b[:, 1],  label="B (zebra)",   color="C2", **kw)
    axes[0].set_title("A vs B")
    axes[0].legend(markerscale=2, fontsize=8)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    # 右: A' vs B
    axes[1].scatter(xy_ap[:, 0], xy_ap[:, 1], label="A' (CycleGAN)", color="C1", **kw)
    axes[1].scatter(xy_b[:, 0],  xy_b[:, 1],  label="B (zebra)",     color="C2", **kw)
    axes[1].set_title("A' vs B")
    axes[1].legend(markerscale=2, fontsize=8)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")

    fig.suptitle("UMAP of train embeddings (A / A' / B jointly fitted)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "umap.png", dpi=150)
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    encoder_name: str = cfg["encoder"]["name"]
    device: str       = cfg["encoder"]["device"]
    data_dir  = Path(cfg["paths"]["data_dir"])
    cache_dir = Path(cfg["paths"]["cache_dir"]) / encoder_name
    out_dir   = Path(cfg["paths"]["output_dir"]) / "preflight" / encoder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    kernels: list[str] = cfg["step0"]["auroc_kernels"]
    seeds = [0, 1, 2, 3, 4]

    emb_a    = np.load(cache_dir / "emb_a.npy")
    emb_ap   = np.load(cache_dir / "emb_a_prime.npy")
    emb_b    = np.load(cache_dir / "emb_b.npy")
    emb_a_te  = np.load(cache_dir / "emb_a_test.npy")
    emb_ap_te = np.load(cache_dir / "emb_a_prime_test.npy")
    emb_b_te  = np.load(cache_dir / "emb_b_test.npy")

    pairs    = pd.read_csv(cfg["paths"]["pairs_csv"])
    pairs_tr = pairs[pairs["split"] == "train"]
    pairs_te = pairs[pairs["split"] == "test"]
    filenames_a_tr = list(pairs_tr["a"])
    filenames_a_te = list(pairs_te["a"])

    assert len(emb_a) == len(emb_ap) == len(filenames_a_tr), "train ペアの行数不一致"
    assert filenames_a_tr == list(pairs_tr["a_prime"]), "a と a_prime のファイル名が不一致"

    delta      = emb_ap - emb_a
    mean_delta = delta.mean(axis=0)
    delta_real = emb_b.mean(axis=0) - emb_a.mean(axis=0)

    metrics: dict = {}

    print("確認1: 変化方向の一致 ...")
    metrics["check1"] = _check1(delta, mean_delta, delta_real, out_dir)

    print("確認2: 識別AUROC ...")
    metrics["check2_auroc"] = _check2(
        emb_a, emb_ap, emb_b, emb_a_te, emb_ap_te, emb_b_te, kernels, seeds
    )

    if encoder_name == "vit_b16":
        print("確認3: シマウマ判定率 ...")
        metrics["check3_zebra"] = _check3(data_dir / "horse2zebra", filenames_a_te, device)
    else:
        print("確認3: スキップ (encoder != vit_b16)")
        metrics["check3_zebra"] = "skipped (encoder != vit_b16)"

    print("確認4: 定数シフト説明率 ...")
    metrics["check4"] = _check4(delta, mean_delta)

    direction_alignment: float = metrics["check4"]["direction_alignment"]
    if direction_alignment < 0.9:
        print("確認5: Δ大小と画像内容の関係 ...")
        _check5(delta, filenames_a_tr, data_dir / "horse2zebra" / "trainA", out_dir)
        metrics["check5_grid"] = str(out_dir / "check5_grid.png")
    else:
        print(f"確認5: スキップ (direction_alignment={direction_alignment:.3f} >= 0.9)")
        metrics["check5_grid"] = f"skipped (direction_alignment={direction_alignment:.3f} >= 0.9)"

    print("UMAP ...")
    _umap_plot(emb_a, emb_ap, emb_b, out_dir)

    out_path = out_dir / "preflight.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
