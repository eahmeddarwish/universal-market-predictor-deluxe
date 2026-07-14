"""
model.py
========
Shared multi-ticker LSTM with learned ticker embeddings + multi-horizon head.

Why this design (vs. one LSTM per ticker)
------------------------------------------
A single backbone trained across the whole universe sees far more market
behavior (crashes, rallies, sideways chop, volatility regimes) than any one
ticker's own history could ever provide, so the shared layers generalize
better. Each ticker still gets its own learned embedding vector so the model
can specialize its output per-asset without needing a separate network per
asset.

Cold-start for new tickers
---------------------------
The embedding table is allocated with MAX_TICKER_SLOTS rows up front (far
more than the current universe size). Adding a new ticker later means
assigning it an unused row index and fine-tuning ONLY that embedding row
(see add_new_ticker.py) with the rest of the network frozen — the new
ticker "joins" the shared system without disturbing anything already
learned for the other tickers, and without a full retrain.

Multi-horizon output
---------------------
Instead of predicting one day ahead and recursively feeding the prediction
back in as input for the next day (which compounds error fast — see the
project's early prototypes), the Dense output layer directly emits one
value per configured horizon (e.g. 1, 3, 7 trading days ahead) from a
single forward pass.

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

MAX_TICKER_SLOTS = 64  # reserved embedding rows — leaves headroom for cold-start additions


def build_shared_model(
    n_features: int,
    look_back: int = 60,
    n_horizons: int = 3,
    embedding_dim: int = 16,
    lstm_units=(128, 64, 32),
    dropout=(0.25, 0.20, 0.15),
    dense_units=(32, 16),
    learning_rate: float = 1e-3,
    loss: str = "huber",
    max_ticker_slots: int = MAX_TICKER_SLOTS,
) -> Model:
    """Build the shared LSTM backbone + ticker-embedding + multi-horizon head."""

    seq_input = layers.Input(shape=(look_back, n_features), name="price_sequence")
    ticker_input = layers.Input(shape=(1,), dtype="int32", name="ticker_id")

    # ── Ticker embedding ──
    ticker_emb = layers.Embedding(
        input_dim=max_ticker_slots,
        output_dim=embedding_dim,
        name="ticker_embedding",
    )(ticker_input)
    ticker_emb = layers.Flatten(name="ticker_embedding_flat")(ticker_emb)

    # ── Shared LSTM backbone ──
    x = seq_input
    for i, (units, drop) in enumerate(zip(lstm_units, dropout)):
        return_seq = i < len(lstm_units) - 1
        x = layers.LSTM(units, return_sequences=return_seq, name=f"lstm_{i+1}")(x)
        x = layers.Dropout(drop, name=f"dropout_{i+1}")(x)
        if return_seq:
            x = layers.BatchNormalization(name=f"batchnorm_{i+1}")(x)

    # ── Merge backbone output with ticker identity ──
    merged = layers.Concatenate(name="merge_backbone_ticker")([x, ticker_emb])

    d = merged
    for i, units in enumerate(dense_units):
        d = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(d)

    output = layers.Dense(n_horizons, activation="linear", name="horizon_output")(d)

    model = Model(inputs=[seq_input, ticker_input], outputs=output, name="universal_market_predictor")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["mae"])
    return model


def freeze_backbone(model: Model, unfreeze_layer_names: list[str] = ("ticker_embedding",)) -> Model:
    """
    Freeze every layer EXCEPT the ones named in `unfreeze_layer_names`
    (used by add_new_ticker.py to fine-tune only a new ticker's embedding
    row, leaving everything the shared backbone learned untouched).
    """
    for layer in model.layers:
        layer.trainable = layer.name in unfreeze_layer_names
    # recompile is required after changing `trainable` flags
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=["mae"])
    return model


def unfreeze_all(model: Model) -> Model:
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=["mae"])
    return model
