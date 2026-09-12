"""ViT-B/16 または OpenCLIP-ViT-H/14 で埋め込みを抽出して cache_dir に保存。

出力:
    <cache_dir>/<encoder>/emb_a.npy       (train A, pairs.csv に従い順序を固定)
    <cache_dir>/<encoder>/emb_a_prime.npy (train A', 同順)
    <cache_dir>/<encoder>/emb_b.npy       (train B)
    <cache_dir>/<encoder>/emb_a_test.npy
    <cache_dir>/<encoder>/emb_b_test.npy
    <cache_dir>/<encoder>/manifest.csv    (split / idx / filename)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import load_config


class _ImageDataset(Dataset[torch.Tensor]):
    def __init__(self, paths: list[Path], transform: object) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)  # type: ignore[operator]


def _build_encoder(name: str, device: torch.device) -> tuple[nn.Module, object]:
    if name == "vit_b16":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        data_cfg = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_cfg, is_training=False)
    elif name == "openclip_vit_h14":
        import open_clip
        model, _, transform = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        model = model.visual
    else:
        raise ValueError(f"未対応のエンコーダ: {name}")
    model.to(device).eval()
    return model, transform


@torch.no_grad()
def _extract(
    paths: list[Path],
    model: nn.Module,
    transform: object,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    ds = _ImageDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    embs: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        embs.append(model(batch.to(device)).cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def main() -> None:
    cfg = load_config()
    encoder_name: str = cfg["encoder"]["name"]
    device = torch.device(cfg["encoder"]["device"] if torch.cuda.is_available() else "cpu")
    data_dir = Path(cfg["paths"]["data_dir"])
    cache_dir = Path(cfg["paths"]["cache_dir"]) / encoder_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_size: int = cfg["train"]["batch_size"]

    model, transform = _build_encoder(encoder_name, device)
    print(f"エンコーダ: {encoder_name}, デバイス: {device}")

    # --- A と A' (pairs.csv の行順を厳守) ---
    pairs = pd.read_csv(cfg["paths"]["pairs_csv"])
    trainA_dir = data_dir / "horse2zebra" / "trainA"
    trainAp_dir = data_dir / "horse2zebra" / "trainA_prime"
    paths_a = [trainA_dir / name for name in pairs["a"]]
    paths_ap = [trainAp_dir / name for name in pairs["a_prime"]]

    emb_a = _extract(paths_a, model, transform, device, batch_size, "emb_a")
    emb_ap = _extract(paths_ap, model, transform, device, batch_size, "emb_a_prime")
    np.save(cache_dir / "emb_a.npy", emb_a)
    np.save(cache_dir / "emb_a_prime.npy", emb_ap)

    # --- B (train) ---
    paths_b = sorted((data_dir / "horse2zebra" / "trainB").glob("*.jpg"))
    emb_b = _extract(paths_b, model, transform, device, batch_size, "emb_b")
    np.save(cache_dir / "emb_b.npy", emb_b)

    # --- A test ---
    paths_a_test = sorted((data_dir / "horse2zebra" / "testA").glob("*.jpg"))
    emb_a_test = _extract(paths_a_test, model, transform, device, batch_size, "emb_a_test")
    np.save(cache_dir / "emb_a_test.npy", emb_a_test)

    # --- B test ---
    paths_b_test = sorted((data_dir / "horse2zebra" / "testB").glob("*.jpg"))
    emb_b_test = _extract(paths_b_test, model, transform, device, batch_size, "emb_b_test")
    np.save(cache_dir / "emb_b_test.npy", emb_b_test)

    # --- manifest.csv ---
    rows = []
    for split, paths in [
        ("a", paths_a), ("a_prime", paths_ap),
        ("b", paths_b), ("a_test", paths_a_test), ("b_test", paths_b_test),
    ]:
        for i, p in enumerate(paths):
            rows.append({"split": split, "idx": i, "filename": p.name})
    pd.DataFrame(rows).to_csv(cache_dir / "manifest.csv", index=False)

    print(f"saved -> {cache_dir}")
    for name, arr in [
        ("emb_a", emb_a), ("emb_a_prime", emb_ap), ("emb_b", emb_b),
        ("emb_a_test", emb_a_test), ("emb_b_test", emb_b_test),
    ]:
        print(f"  {name}: {arr.shape}")


if __name__ == "__main__":
    main()
