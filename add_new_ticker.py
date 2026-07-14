"""
add_new_ticker.py
==================
Cold-start a brand-new ticker (new stock, new exchange, new coin) into the
ALREADY-TRAINED shared model — without retraining the whole system.

How it works
------------
1. The shared backbone (LSTM stack + Dense head) is frozen — its weights
   don't move.
2. Only the ticker embedding table is left trainable. This sounds like it
   would risk disturbing every other ticker's embedding row too, since it's
   one shared weight matrix — but it doesn't: Keras Embedding layers produce
   SPARSE gradients, meaning only the row(s) actually looked up during a
   given batch receive a gradient update. Since every training batch here
   only ever contains the new ticker's id, only its row moves. Every other
   ticker's embedding is mathematically untouched.
3. A small number of fine-tuning epochs (config.yaml -> fine_tune.epochs)
   trains just that one new row to a sensible position in embedding space,
   informed by the shared backbone's general market knowledge.
4. The new ticker gets folded into ticker_id_map.json, gets its own scaler,
   and the model is saved back — ready to serve predictions for it
   immediately, with no impact on any ticker that was already live.

Usage
-----
    python add_new_ticker.py --ticker 2010.SR --name "SABIC" --region SA

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import registry
from src.data_fetcher import fetch_ohlcv
from src.indicators import add_all_indicators
from src.dataset import build_ticker_dataset
from src.model import freeze_backbone, unfreeze_all, MAX_TICKER_SLOTS
from src.evaluation import evaluate_ticker


def main():
    parser = argparse.ArgumentParser(description="Fold a new ticker into the shared model.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--region", default="?")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--registry-dir", default=registry.DEFAULT_REGISTRY_DIR)
    args = parser.parse_args()

    cfg = registry.load_config(args.config)
    ticker = args.ticker.upper().strip()

    model_path = os.path.join(args.registry_dir, "shared_model.keras")
    if not os.path.exists(model_path):
        print(f"❌ No trained model found at '{model_path}'. Run train.py first.")
        return

    ticker_id_map = registry.load_ticker_id_map(args.registry_dir)
    meta = registry.load_meta(args.registry_dir)

    if ticker in ticker_id_map:
        print(f"⚠️  '{ticker}' already has embedding id {ticker_id_map[ticker]}. "
              f"Re-running will just re-fine-tune its existing row.")

    if len(ticker_id_map) >= MAX_TICKER_SLOTS and ticker not in ticker_id_map:
        print(f"❌ Embedding table is full ({MAX_TICKER_SLOTS} slots used). "
              f"Increase MAX_TICKER_SLOTS in src/model.py and retrain from scratch "
              f"(train.py) to grow it.")
        return

    # ── Assign (or reuse) an embedding row for this ticker ──
    new_map = registry.assign_ticker_ids([{"ticker": ticker}], ticker_id_map)
    new_id = new_map[ticker]
    print(f"\nTicker '{ticker}' -> embedding row {new_id}")

    # ── Fetch + engineer features for just this ticker ──
    print(f"Fetching data for {ticker} ...")
    fr = fetch_ohlcv(ticker, period=cfg["data"]["history_period"])
    if fr["error"] or fr["df"] is None:
        print(f"❌ {fr['error']}")
        return
    if len(fr["df"]) < cfg["data"]["min_rows_required"]:
        print(f"❌ Only {len(fr['df'])} rows of history — need at least "
              f"{cfg['data']['min_rows_required']}. Cannot fine-tune reliably.")
        return

    df_ind = add_all_indicators(fr["df"])
    ds = build_ticker_dataset(
        df_ind, ticker, new_id,
        look_back=meta.get("look_back", cfg["model"]["look_back"]),
        horizons=meta.get("horizons", cfg["model"]["horizons"]),
        train_fraction=cfg["data"]["train_fraction"],
    )
    if ds is None:
        print("❌ Not enough usable rows after indicator warm-up to build train/test windows.")
        return

    # ── Load shared model, freeze everything except the embedding table ──
    print("\nLoading shared model and freezing backbone ...")
    model = tf.keras.models.load_model(model_path)
    model = freeze_backbone(model, unfreeze_layer_names=["ticker_embedding"])

    # Recompile with a smaller learning rate for gentle fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["fine_tune"]["learning_rate"]),
        loss=model.loss,
        metrics=["mae"],
    )

    train_ids = np.full((len(ds.X_train), 1), new_id)
    test_ids = np.full((len(ds.X_test), 1), new_id)

    print(f"\nFine-tuning embedding for '{ticker}' "
          f"({cfg['fine_tune']['epochs']} epochs, backbone frozen) ...")
    model.fit(
        x=[ds.X_train, train_ids],
        y=ds.y_train,
        validation_data=([ds.X_test, test_ids], ds.y_test),
        epochs=cfg["fine_tune"]["epochs"],
        batch_size=min(32, max(4, len(ds.X_train) // 4)),
        verbose=2,
    )

    # ── Evaluate the new ticker vs. baselines before we commit it ──
    pred_returns = model.predict([ds.X_test, test_ids], verbose=0)
    report = evaluate_ticker(ds, df_ind, pred_returns, baselines=cfg["evaluation"]["baselines"])
    print(f"\n--- {ticker} vs. baselines ---")
    print(report.to_string(index=False))

    # ── Commit: unfreeze for hygiene, save model + registry updates ──
    model = unfreeze_all(model)
    model.save(model_path)
    registry.save_ticker_id_map(new_map, args.registry_dir)
    registry.save_scaler(ticker, ds.scaler, args.registry_dir)

    trained = set(meta.get("trained_tickers", []))
    trained.add(ticker)
    meta["trained_tickers"] = sorted(trained)
    registry.save_meta(meta, args.registry_dir)

    print(f"\n✅ '{ticker}' is now part of the shared model (embedding row {new_id}).")
    print(f"   Remember to also add it to config.yaml's `universe` list so the next "
          f"full retrain (train.py) includes it from scratch too.")


if __name__ == "__main__":
    main()
