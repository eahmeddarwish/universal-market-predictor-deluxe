"""
registry.py
===========
Everything needed to load/save the shared model "as a system": the
ticker <-> embedding-row-id map, per-ticker scalers, config, and metadata.
Used by train.py, add_new_ticker.py, and app.py so all three always agree
on which ticker maps to which embedding row and which scaler to use.

Layout on disk (model_registry/):
    shared_model.keras       final trained Keras model
    ticker_id_map.json       {"AAPL": 0, "MSFT": 1, ...}
    meta.json                feature_cols, look_back, horizons, region map
    scalers/<TICKER>.pkl     one fitted MinMaxScaler per ticker
    metrics/*.csv            evaluation reports

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import os
import json
import pickle
import yaml

DEFAULT_REGISTRY_DIR = "model_registry"


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def full_universe_list(config: dict) -> list[dict]:
    """Flatten config['universe']['stocks'] + ['crypto'] into one list of dicts."""
    u = config["universe"]
    return list(u.get("stocks", [])) + list(u.get("crypto", []))


def load_ticker_id_map(registry_dir: str = DEFAULT_REGISTRY_DIR) -> dict:
    path = os.path.join(registry_dir, "ticker_id_map.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ticker_id_map(mapping: dict, registry_dir: str = DEFAULT_REGISTRY_DIR):
    os.makedirs(registry_dir, exist_ok=True)
    path = os.path.join(registry_dir, "ticker_id_map.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def assign_ticker_ids(universe_entries: list[dict], existing_map: dict | None = None) -> dict:
    """
    Give every ticker a stable integer id. Existing tickers KEEP their id
    (critical — changing an id after training would silently corrupt that
    ticker's learned embedding). New tickers get the next free id.
    """
    mapping = dict(existing_map or {})
    used_ids = set(mapping.values())
    next_id = 0

    def _next_free():
        nonlocal next_id
        while next_id in used_ids:
            next_id += 1
        used_ids.add(next_id)
        return next_id

    for entry in universe_entries:
        tk = entry["ticker"]
        if tk not in mapping:
            mapping[tk] = _next_free()

    return mapping


def save_scaler(ticker: str, scaler, registry_dir: str = DEFAULT_REGISTRY_DIR):
    d = os.path.join(registry_dir, "scalers")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{_safe_name(ticker)}.pkl"), "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(ticker: str, registry_dir: str = DEFAULT_REGISTRY_DIR):
    path = os.path.join(registry_dir, "scalers", f"{_safe_name(ticker)}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_meta(meta: dict, registry_dir: str = DEFAULT_REGISTRY_DIR):
    os.makedirs(registry_dir, exist_ok=True)
    with open(os.path.join(registry_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_meta(registry_dir: str = DEFAULT_REGISTRY_DIR) -> dict:
    path = os.path.join(registry_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_name(ticker: str) -> str:
    return ticker.replace("/", "_").replace("^", "_").replace(".", "_")
