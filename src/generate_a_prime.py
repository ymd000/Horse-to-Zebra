"""CycleGAN(馬→シマウマ)で A から A' を生成し、pairs.csv を書き出す。

出力:
    data/horse2zebra/trainA_prime/*.jpg
    data/pairs.csv  (columns: a, a_prime, flip)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_config


def main() -> None:
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    src_dir = data_dir / "horse2zebra" / "trainA"
    dst_dir = data_dir / "horse2zebra" / "trainA_prime"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # TODO: CycleGAN 生成器をロード (data/cyclegan_weights/horse2zebra.pth)
    # TODO: src_dir の各画像を推論し dst_dir に保存 (256x256, 命名は元と同じ)

    rows = []
    for a_path in sorted(src_dir.glob("*.jpg")):
        ap_path = dst_dir / a_path.name
        rows.append({"a": a_path.name, "a_prime": ap_path.name, "flip": False})
        if cfg["data"]["flip_augment"]:
            rows.append({"a": a_path.name, "a_prime": ap_path.name, "flip": True})

    pairs_csv = Path(cfg["paths"]["pairs_csv"])
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(pairs_csv, index=False)
    print(f"pairs.csv: {len(rows)} 行 -> {pairs_csv}")


if __name__ == "__main__":
    main()
