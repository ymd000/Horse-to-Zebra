"""config.yaml の読み込みと CLI 引数によるドット記法の上書き。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _coerce(raw: str) -> Any:
    # YAMLのスカラ解釈に委ねる（"true"→True, "1e-3"→float, "null"→None など）
    return yaml.safe_load(raw)


def load_config(argv: list[str] | None = None) -> dict:
    """--config で config.yaml を、--set key.path=value で個別上書き。

    例:
        uv run python src/train.py --set sae.n_site_latent=4 --set train.lam_l1=1e-4
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    known, _ = parser.parse_known_args(argv)

    with open(known.config) as f:
        cfg = yaml.safe_load(f)

    for override in known.set:
        if "=" not in override:
            raise ValueError(f"--set は KEY=VALUE 形式: {override}")
        key, raw = override.split("=", 1)
        _set_dotted(cfg, key, _coerce(raw))

    return cfg


def run_tag(cfg: dict) -> str:
    """outputs/<tag>/ に使う短い実験タグ。主要ハイパラをファイル名安全な形で。"""
    s = cfg["sae"]
    t = cfg["train"]
    return f"s{s['n_site_latent']}_exp{s['expansion']}_l1{t['lam_l1']:.0e}_pair{t['lam_pair']}"
