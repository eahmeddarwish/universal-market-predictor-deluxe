"""
train.py
========
Full training pipeline for the shared multi-ticker Universal Market
Predictor. Designed to be resumable, so a Colab disconnect never means
starting over from epoch 0.

Usage
-----
Local:
    python train.py
    python train.py --config config.yaml --registry-dir model_registry

Google Colab (recommended — free GPU):
    !git clone <your-repo-url>
    %cd universal-market-predictor-deluxe
    !pip install -r requirements.txt -q
    from google.colab import drive
    drive.mount('/content/drive')
    !python train.py

Resuming after a disconnect
----------------------------
Every `checkpoint_every_n_epochs` epochs (see config.yaml), the model AND a
small progress.json (last_completed_epoch) are saved to
config["training"]["drive_checkpoint_dir"] when running on Colab (so the
checkpoint survives even if the VM is recycled), or to
config["training"]["local_checkpoint_dir"] otherwise. Simply re-run
`python train.py` — it detects the checkpoint and continues from
last_completed_epoch + 1 instead of retraining from scratch.

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import registry
from src.data_fetcher import fetch_universe
from src.indicators import add_all_indicators
from src.dataset import build_ticker_dataset, assemble_universe
from src.model import build_shared_model
from src.evaluation import evaluate_ticker, walk_forward_folds


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _checkpoint_dir(cfg: dict) -> str:
    d = cfg["training"]["drive_checkpoint_dir"] if _in_colab() else cfg["training"]["local_checkpoint_dir"]
    os.makedirs(d, exist_ok=True)
    return d


class EpochCheckpoint(Callback):
    """Saves the full model + a progress marker every N epochs so training
    can resume exactly where it left off after a Colab disconnect."""

    def __init__(self, ckpt_dir: str, every_n: int = 1):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.every_n = max(1, every_n)
        self.model_path = os.path.join(ckpt_dir, "in_progress_model.keras")
        self.progress_path = os.path.join(ckpt_dir, "progress.json")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            self.model.save(self.model_path)
            with open(self.progress_path, "w") as f:
                json.dump({
                    "last_completed_epoch": epoch,
                    "val_loss": float(logs.get("val_loss", -1)) if logs else -1,
                }, f)
            print(f"  [checkpoint] saved at epoch {epoch + 1} -> {self.model_path}")


def _load_resume_state(ckpt_dir: str):
    progress_path = os.path.join(ckpt_dir, "progress.json")
    model_path = os.path.join(ckpt_dir, "in_progress_model.keras")
    if os.path.exists(progress_path) and os.path.exists(model_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
        model = tf.keras.models.load_model(model_path)
        return model, progress["last_completed_epoch"] + 1
    return None, 0


def build_all_datasets(cfg: dict, ticker_id_map: dict, verbose: bool = True):
    """Fetch + engineer features + window every ticker in the universe.

    Returns (per_ticker_datasets: dict[str, TickerDataset],
             per_ticker_df_ind:  dict[str, pd.DataFrame])
    Tickers with insufficient history are skipped and reported, not silently
    dropped.
    """
    entries = registry.full_universe_list(cfg)
    tickers = [e["ticker"] for e in entries]

    raw = fetch_universe(tickers, period=cfg["data"]["history_period"], verbose=verbose)

    per_ticker_ds, per_ticker_df = {}, {}
    skipped = []

    for entry in entries:
        tk = entry["ticker"]
        fr = raw[tk]
        if fr["error"] or fr["df"] is None or len(fr["df"]) < cfg["data"]["min_rows_required"]:
            skipped.append(tk)
            continue

        df_ind = add_all_indicators(fr["df"])
        ds = build_ticker_dataset(
            df_ind, tk, ticker_id_map[tk],
            look_back=cfg["model"]["look_back"],
            horizons=cfg["model"]["horizons"],
            train_fraction=cfg["data"]["train_fraction"],
        )
        if ds is None:
            skipped.append(tk)
            continue

        per_ticker_ds[tk] = ds
        per_ticker_df[tk] = df_ind

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} ticker(s) (insufficient history): {skipped}\n")

    return per_ticker_ds, per_ticker_df


def main():
    parser = argparse.ArgumentParser(description="Train the shared Universal Market Predictor model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--registry-dir", default=registry.DEFAULT_REGISTRY_DIR)
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing checkpoint and start from epoch 0.")
    args = parser.parse_args()

    cfg = registry.load_config(args.config)
    ckpt_dir = _checkpoint_dir(cfg)

    print("=" * 70)
    print("Universal Market Predictor — Deluxe — Training")
    print(f"Running on Colab: {_in_colab()}")
    print(f"Checkpoint dir  : {ckpt_dir}")
    print("=" * 70)

    # ── 1. Ticker id assignment (stable across runs) ──
    existing_map = registry.load_ticker_id_map(args.registry_dir)
    entries = registry.full_universe_list(cfg)
    ticker_id_map = registry.assign_ticker_ids(entries, existing_map)
    registry.save_ticker_id_map(ticker_id_map, args.registry_dir)

    # ── 2. Data + features + windows for every ticker ──
    print("\n--- Fetching data & building datasets ---")
    per_ticker_ds, per_ticker_df = build_all_datasets(cfg, ticker_id_map)
    if not per_ticker_ds:
        print("No tickers had enough data to train on. Aborting.")
        return

    # ── 3. Assemble the combined multi-ticker training tensors ──
    combined = assemble_universe(per_ticker_ds)
    n_features = combined["X_train"].shape[2]
    n_horizons = len(cfg["model"]["horizons"])
    print(f"\nTrainable tickers: {len(per_ticker_ds)}")
    print(f"Total training windows: {len(combined['X_train'])}  |  test windows: {len(combined['X_test'])}")
    print(f"Features per timestep: {n_features}  |  forecast horizons: {cfg['model']['horizons']}")

    # ── 4. Resume-or-build the shared model ──
    model, initial_epoch = (None, 0)
    if not args.fresh:
        model, initial_epoch = _load_resume_state(ckpt_dir)

    if model is None:
        model = build_shared_model(
            n_features=n_features,
            look_back=cfg["model"]["look_back"],
            n_horizons=n_horizons,
            embedding_dim=cfg["model"]["embedding_dim"],
            lstm_units=cfg["model"]["lstm_units"],
            dropout=cfg["model"]["dropout"],
            dense_units=cfg["model"]["dense_units"],
            learning_rate=cfg["model"]["learning_rate"],
            loss=cfg["model"]["loss"],
        )
        print("\nBuilt a fresh shared model.")
    else:
        print(f"\nResumed from checkpoint at epoch {initial_epoch}.")

    model.summary()

    # ── 5. Train (resumable) ──
    total_epochs = cfg["training"]["epochs"]
    if initial_epoch >= total_epochs:
        print("Checkpoint already reached the configured epoch count — nothing to train.")
    else:
        callbacks = [
            EpochCheckpoint(ckpt_dir, every_n=cfg["training"]["checkpoint_every_n_epochs"]),
            EarlyStopping(monitor="val_loss", patience=cfg["training"]["early_stopping_patience"], restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg["training"]["reduce_lr_patience"], min_lr=1e-6),
        ]

        model.fit(
            x=[combined["X_train"], combined["ticker_ids_train"]],
            y=combined["y_train"],
            validation_data=(
                [combined["X_test"], combined["ticker_ids_test"]],
                combined["y_test"],
            ),
            epochs=total_epochs,
            initial_epoch=initial_epoch,
            batch_size=cfg["training"]["batch_size"],
            callbacks=callbacks,
            verbose=2,
        )

    # ── 6. Evaluate every ticker: model vs baselines, + walk-forward ──
    print("\n--- Evaluating (model vs. honest baselines) ---")
    comparison_frames, walk_frames = [], []

    for tk, ds in per_ticker_ds.items():
        ticker_ids_arr = np.full((len(ds.X_test), 1), ds.ticker_id)
        pred_returns = model.predict([ds.X_test, ticker_ids_arr], verbose=0)

        comparison_frames.append(
            evaluate_ticker(ds, per_ticker_df[tk], pred_returns, baselines=cfg["evaluation"]["baselines"])
        )
        walk_frames.append(
            walk_forward_folds(ds, per_ticker_df[tk], pred_returns, n_folds=cfg["evaluation"]["walk_forward_folds"])
        )

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    walk_df = pd.concat(walk_frames, ignore_index=True)

    metrics_dir = os.path.join(args.registry_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(metrics_dir, "comparison_report.csv"), index=False)
    walk_df.to_csv(os.path.join(metrics_dir, "walk_forward_report.csv"), index=False)

    print("\nModel vs. baseline (first rows):")
    print(comparison_df.head(20).to_string(index=False))

    # ── 7. Save final artifacts ──
    os.makedirs(args.registry_dir, exist_ok=True)
    model.save(os.path.join(args.registry_dir, "shared_model.keras"))
    for tk, ds in per_ticker_ds.items():
        registry.save_scaler(tk, ds.scaler, args.registry_dir)

    registry.save_meta({
        "feature_cols": next(iter(per_ticker_ds.values())).feature_cols,
        "look_back": cfg["model"]["look_back"],
        "horizons": cfg["model"]["horizons"],
        "trained_tickers": list(per_ticker_ds.keys()),
        "skipped_tickers": [e["ticker"] for e in entries if e["ticker"] not in per_ticker_ds],
    }, args.registry_dir)

    print(f"\n✅ Done. Artifacts saved to '{args.registry_dir}/'.")
    print("   - shared_model.keras")
    print("   - ticker_id_map.json, meta.json")
    print("   - scalers/<TICKER>.pkl")
    print("   - metrics/comparison_report.csv, metrics/walk_forward_report.csv")


if __name__ == "__main__":
    main()
