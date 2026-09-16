"""B (trainB / testB) の JPEG を PNG に変換して保存する。

A' は CycleGAN 出力を PNG で保存しており、B との経路対称性
(uint8 sRGB PNG から encoder に入る) を確保するための前処理。
元の JPEG は残し、変換先を trainB_png / testB_png に置く。

出力:
    data/horse2zebra/trainB_png/*.png
    data/horse2zebra/testB_png/*.png
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import load_config


def _convert(src_dir: Path, dst_dir: Path, desc: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in tqdm(sorted(src_dir.glob("*.jpg")), desc=desc):
        img = Image.open(src).convert("RGB")
        img.save(dst_dir / (src.stem + ".png"), format="PNG")


def main() -> None:
    cfg = load_config()
    h2z = Path(cfg["paths"]["data_dir"]) / "horse2zebra"
    _convert(h2z / "trainB", h2z / "trainB_png", "train B -> png")
    _convert(h2z / "testB",  h2z / "testB_png",  "test B -> png")


if __name__ == "__main__":
    main()
