"""CycleGAN(馬→シマウマ)で A から A' を生成し、pairs.csv を書き出す。

A' は PNG (無損失, uint8) で保存する。JPEG 再圧縮による高周波アーティファクトが
B との比較にバイアスを持ち込むのを避けるため。B と同じ uint8 sRGB になるので
encoder への入力段階では非対称性はない。

出力:
    data/horse2zebra/trainA_prime/*.png
    data/horse2zebra/testA_prime/*.png
    data/pairs.csv  (columns: split, a, a_prime)  a は .jpg、a_prime は .png
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import load_config


# 公式の学習済み重みは InstanceNorm2d(track_running_stats=True) で保存されている
_NORM = functools.partial(nn.InstanceNorm2d, track_running_stats=True)


class _ResnetBlock(nn.Module):
    def __init__(self, dim: int, use_dropout: bool) -> None:
        super().__init__()
        use_bias = True
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias),
            _NORM(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias),
            _NORM(dim),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class _ResnetGenerator(nn.Module):
    """junyanz/pytorch-CycleGAN-and-pix2pix の ResnetGenerator と互換。"""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()
        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=True),
            _NORM(ngf),
            nn.ReLU(True),
        ]
        n_down = 2
        for i in range(n_down):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=True),
                _NORM(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2**n_down
        for _ in range(n_blocks):
            model.append(_ResnetBlock(ngf * mult, use_dropout=False))
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2, 3,
                    stride=2, padding=1, output_padding=1, bias=True,
                ),
                _NORM(ngf * mult // 2),
                nn.ReLU(True),
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _load_generator(weights_path: Path, device: torch.device) -> _ResnetGenerator:
    net = _ResnetGenerator()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


# horse2zebra は元から 256x256。encoder への入力段階で A' が B に対して
# 余分な Resize を経由しないよう、ここでは Resize/CenterCrop を掛けない
# (掛けなくても 256x256 のままなので実質等価だが、非 256 入力を静かに
# 通してしまうリスクを排除する)。
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _tensor_to_image(t: torch.Tensor) -> Image.Image:
    arr = t.cpu().float().numpy()
    arr = (arr * 0.5 + 0.5).clip(0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _generate(
    net: _ResnetGenerator, src_dir: Path, dst_dir: Path, device: torch.device, desc: str
) -> list[tuple[str, str]]:
    """src_dir の .jpg を推論し dst_dir に PNG で保存、(a名.jpg, a'名.png) の列を返す。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    with torch.no_grad():
        for a_path in tqdm(sorted(src_dir.glob("*.jpg")), desc=desc):
            img = Image.open(a_path).convert("RGB")
            if img.size != (256, 256):
                raise ValueError(f"{a_path}: 期待サイズ (256, 256) と異なる: {img.size}")
            x = _TRANSFORM(img).unsqueeze(0).to(device)
            out = net(x).squeeze(0)
            out_name = a_path.stem + ".png"
            _tensor_to_image(out).save(dst_dir / out_name, format="PNG")
            pairs.append((a_path.name, out_name))
    return pairs


def main() -> None:
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    h2z = data_dir / "horse2zebra"

    weights_path = data_dir / "cyclegan_weights" / "horse2zebra.pth"
    device = torch.device(cfg["encoder"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    net = _load_generator(weights_path, device)
    print(f"CycleGAN ジェネレータをロード: {weights_path}")

    train_pairs = _generate(net, h2z / "trainA", h2z / "trainA_prime", device, "train A'")
    test_pairs  = _generate(net, h2z / "testA",  h2z / "testA_prime",  device, "test A'")

    rows = (
        [{"split": "train", "a": a, "a_prime": ap} for a, ap in train_pairs]
        + [{"split": "test",  "a": a, "a_prime": ap} for a, ap in test_pairs]
    )
    pairs_csv = Path(cfg["paths"]["pairs_csv"])
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(pairs_csv, index=False)
    print(f"pairs.csv: {len(rows)} 行 -> {pairs_csv}")


if __name__ == "__main__":
    main()
