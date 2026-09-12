#!/usr/bin/env bash
# horse2zebra データセット + 学習済み CycleGAN 重みを取得
# 使い方: bash data/download.sh
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DATA_DIR"

# --- horse2zebra ---
# junyanz/pytorch-CycleGAN-and-pix2pix の公式配布
if [ ! -d horse2zebra ]; then
  echo "[1/2] horse2zebra を取得..."
  curl -L -o horse2zebra.zip \
    https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip
  unzip -q horse2zebra.zip
  rm horse2zebra.zip
else
  echo "[1/2] horse2zebra は既に存在: skip"
fi

# --- 学習済み CycleGAN (horse -> zebra) ---
# 参考: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh
if [ ! -f cyclegan_weights/horse2zebra.pth ]; then
  echo "[2/2] CycleGAN horse2zebra 重みを取得..."
  mkdir -p cyclegan_weights
  curl -L -o cyclegan_weights/horse2zebra.pth \
    http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/horse2zebra.pth
else
  echo "[2/2] CycleGAN 重みは既に存在: skip"
fi

echo "完了。次に: uv run python src/generate_a_prime.py"
