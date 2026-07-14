"""
dataset.py
==========
Turns per-ticker OHLCV+indicator dataframes into the (X, ticker_id, y) tensors
the shared multi-ticker model trains on.

Design notes
------------
* Every ticker gets its OWN MinMaxScaler, fit ONLY on that ticker's training
  window (no look-ahead leakage, and no cross-ticker leakage either -- an
  Aramco share priced in SAR and Bitcoin priced in USD have wildly different
  scales, so per-ticker scaling is mandatory, not optional). The scaler is
  fit on FEATURE_COLS only, and every one of those is already a bounded
  ratio/percentage (see src/indicators.py) -- never a raw price level -- so
  it doesn't need to extrapolate past what it saw during training even for
  a strongly trending ticker.
* The LSTM *backbone* is shared across all tickers. What tells the network
  "this window belongs to AAPL, not BTC" is the integer `ticker_id`, which
  gets embedded (see src/model.py) and concatenated into the network.
* Targets are multi-horizon PERCENTAGE RETURNS, not scaled price levels:
  for each window, y[h] = (Close[i+h-1] - Close[i-1]) / Close[i-1], for
  every h in config["model"]["horizons"] (default [1, 3, 7]) -- a single
  forward pass produces all horizons at once, instead of the old recursive
  "feed the prediction back in" approach that compounds error day over day.
  Predicting a return instead of a raw (scaled) price was a deliberate fix:
  an earlier version of this project predicted a MinMax-scaled Close price,
  and a real training run showed it losing badly to a naive baseline on
  strongly-trending stocks -- because the test period's prices sat outside
  the [0,1] range the scaler ever saw during training, the model was being
  asked to extrapolate blind. A percentage return stays in a similar,
  bounded numeric range regardless of how much the absolute price has grown,
  so there's nothing to extrapolate.

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

from src.indicators import FEATURE_COLS


@dataclass
class TickerDataset:
    ticker: str
    ticker_id: int
    scaler: MinMaxScaler
    feature_cols: list
    X_train: np.ndarray
    y_train: np.ndarray          # shape (n, n_horizons)
    X_test: np.ndarray
    y_test: np.ndarray
    dates_test: pd.DatetimeIndex
    raw_close_test: np.ndarray     # shape (n, n_horizons) -- actual unscaled close at each horizon target
    anchor_close_test: np.ndarray  # shape (n,) -- actual unscaled close on the "today" day the window ends on
    horizons: list


def build_ticker_dataset(
    df_ind: pd.DataFrame,
    ticker: str,
    ticker_id: int,
    look_back: int = 60,
    horizons: list[int] = (1, 3, 7),
    train_fraction: float = 0.80,
) -> TickerDataset | None:
    """
    Build sliding-window (X, y) sequences for a single ticker.

    Returns None if there isn't enough history to build at least a handful
    of train + test windows (caller should skip/log that ticker).
    """
    available = [c for c in FEATURE_COLS if c in df_ind.columns]
    data = df_ind[available].values
    max_h = max(horizons)

    n_samples = len(data)
    total_windows = n_samples - look_back - max_h + 1
    if total_windows < 30:
        return None

    train_windows = int(total_windows * train_fraction)
    # last raw row touched by any TRAINING window (input or furthest target)
    raw_train_end = look_back + train_windows + max_h - 1

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:raw_train_end])       # fit on training period only
    scaled = scaler.transform(data)         # apply to the whole series

    close_raw = df_ind["Close"].values
    X, y, target_dates, raw_close_targets, anchor_close = [], [], [], [], []
    for i in range(look_back, n_samples - max_h + 1):
        anchor_price = close_raw[i - 1]     # actual price on "today", the window's last input day
        X.append(scaled[i - look_back:i])
        y.append([(close_raw[i + h - 1] - anchor_price) / anchor_price for h in horizons])  # % return, not scaled price
        target_dates.append(df_ind.index[i - 1])       # anchor date = "today", the window's last input day
        raw_close_targets.append([close_raw[i + h - 1] for h in horizons])
        anchor_close.append(anchor_price)

    X = np.array(X)
    y = np.array(y)
    target_dates = pd.DatetimeIndex(target_dates)
    raw_close_targets = np.array(raw_close_targets)      # (n, n_horizons)
    anchor_close = np.array(anchor_close)                 # (n,)

    X_train, y_train = X[:train_windows], y[:train_windows]
    X_test, y_test = X[train_windows:], y[train_windows:]
    dates_test = target_dates[train_windows:]
    raw_close_test = raw_close_targets[train_windows:]
    anchor_close_test = anchor_close[train_windows:]

    if len(X_test) < 5:
        return None

    return TickerDataset(
        ticker=ticker,
        ticker_id=ticker_id,
        scaler=scaler,
        feature_cols=available,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        dates_test=dates_test,
        raw_close_test=raw_close_test,
        anchor_close_test=anchor_close_test,
        horizons=list(horizons),
    )


def return_to_price(anchor_price: np.ndarray, predicted_return: np.ndarray) -> np.ndarray:
    """
    Convert the model's predicted percentage return back to a price:
        price = anchor_price * (1 + predicted_return)
    Replaces the old scaler-based inverse_close() now that targets are
    returns rather than scaled Close values -- no scaler involved at all,
    which is exactly what avoids the extrapolation problem for trending
    tickers (see module docstring).
    """
    return np.asarray(anchor_price) * (1.0 + np.asarray(predicted_return))


def assemble_universe(
    per_ticker_datasets: dict[str, TickerDataset],
) -> dict:
    """
    Stack every ticker's train/test windows into single arrays the shared
    model can train on in one pass, tagging each row with its ticker_id.

    Returns dict with keys:
        X_train, ticker_ids_train, y_train,
        X_test,  ticker_ids_test,  y_test
    """
    X_train_list, id_train_list, y_train_list = [], [], []
    X_test_list, id_test_list, y_test_list = [], [], []

    for ds in per_ticker_datasets.values():
        X_train_list.append(ds.X_train)
        id_train_list.append(np.full(len(ds.X_train), ds.ticker_id))
        y_train_list.append(ds.y_train)

        X_test_list.append(ds.X_test)
        id_test_list.append(np.full(len(ds.X_test), ds.ticker_id))
        y_test_list.append(ds.y_test)

    return {
        "X_train": np.concatenate(X_train_list, axis=0),
        "ticker_ids_train": np.concatenate(id_train_list, axis=0),
        "y_train": np.concatenate(y_train_list, axis=0),
        "X_test": np.concatenate(X_test_list, axis=0),
        "ticker_ids_test": np.concatenate(id_test_list, axis=0),
        "y_test": np.concatenate(y_test_list, axis=0),
    }
