"""ViT-B/16 または OpenCLIP-ViT-H/14 で埋め込みを抽出して cache_dir に保存。

出力:
    <cache_dir>/<encoder>/emb_a.npy       (train A, ペアCSVに従い順序を固定)
    <cache_dir>/<encoder>/emb_a_prime.npy (train A', 同順)
    <cache_dir>/<encoder>/emb_b.npy       (train B)
    <cache_dir>/<encoder>/emb_a_test.npy
    <cache_dir>/<encoder>/emb_b_test.npy
    <cache_dir>/<encoder>/manifest.csv    (どのファイルがどの行か)
"""
from __future__ import annotations

from pathlib import Path

from config import load_config


def main() -> None:
    cfg = load_config()
    encoder = cfg["encoder"]["name"]
    cache_dir = Path(cfg["paths"]["cache_dir"]) / encoder
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TODO: encoder 名に応じてモデルをロード
    #   vit_b16       -> timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    #   openclip_vit_h14 -> open_clip.create_model('ViT-H-14', pretrained='laion2b_s32b_b79k') の visual
    # TODO: pairs.csv を読み、flip を反映して A と A' の埋め込みを同順に抽出
    # TODO: B, A_test, B_test も抽出
    # TODO: np.save で保存、manifest.csv も書き出し

    print(f"saved -> {cache_dir}")


if __name__ == "__main__":
    main()
