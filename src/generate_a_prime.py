"""CycleGAN(馬→シマウマ)で A から A' を生成し、pairs.csv を書き出す。

出力:
    data/horse2zebra/trainA_prime/*.jpg
    data/pairs.csv  (columns: a, a_prime, flip)
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


class _ResnetBlock(nn.Module):
    def __init__(self, dim: int, norm_layer: type, use_dropout: bool) -> None:
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias),
            norm_layer(dim),
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
        norm_layer = nn.InstanceNorm2d
        use_bias = True  # InstanceNorm では bias を使う

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_down = 2
        for i in range(n_down):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2**n_down
        for _ in range(n_blocks):
            model.append(_ResnetBlock(ngf * mult, norm_layer, use_dropout=False))
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2, 3,
                    stride=2, padding=1, output_padding=1, bias=use_bias,
                ),
                norm_layer(ngf * mult // 2),
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


_TRANSFORM = transforms.Compose([
    transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _tensor_to_image(t: torch.Tensor) -> Image.Image:
    arr = t.cpu().float().numpy()
    arr = (arr * 0.5 + 0.5).clip(0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    src_dir = data_dir / "horse2zebra" / "trainA"
    dst_dir = data_dir / "horse2zebra" / "trainA_prime"
    dst_dir.mkdir(parents=True, exist_ok=True)

    weights_path = data_dir / "cyclegan_weights" / "horse2zebra.pth"
    device = torch.device(cfg["encoder"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    net = _load_generator(weights_path, device)
    print(f"CycleGAN ジェネレータをロード: {weights_path}")

    img_paths = sorted(src_dir.glob("*.jpg"))
    with torch.no_grad():
        for a_path in tqdm(img_paths, desc="generating A'"):
            img = Image.open(a_path).convert("RGB")
            x = _TRANSFORM(img).unsqueeze(0).to(device)
            out = net(x).squeeze(0)
            _tensor_to_image(out).save(dst_dir / a_path.name, quality=95)

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
