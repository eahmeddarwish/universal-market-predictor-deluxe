"""
indicators.py
=============
Pure pandas/numpy technical indicators (no ta-lib dependency, so it installs
cleanly on Colab, HF Spaces, and any local machine without compiler headaches).

12 features, ALL scale/trend-invariant (bounded ratios or percentages, never
a raw price or volume level) -- matching the shared model's expected
FEATURE_COLS:
    RSI, MACD_pct, MACD_Signal_pct, BB_Upper_pct, BB_Lower_pct, BB_Width,
    ATR_pct, Daily_Return, Volatility_10d, Volume_Ratio,
    Price_vs_MA20, Price_vs_MA50

Why "_pct" versions instead of raw MACD/BB/ATR/MA levels
----------------------------------------------------------
A real training run on this project surfaced the problem directly: stocks
like AAPL/MSFT that trended strongly upward over the 10-year training window
pushed the TEST period's raw price-based indicators (MACD, Bollinger Bands,
moving averages, ATR -- all absolute price units) far outside the [0,1] range
a MinMaxScaler fit only on the training period ever saw. The model was
effectively asked to extrapolate on inputs it had never encountered, and it
failed badly (worse than a naive "tomorrow = today" baseline). Expressing
every price-derived indicator as a ratio/percentage of the current price
keeps it in a similar numeric range regardless of how much the absolute
price level has grown by the test period, which is what actually fixes this
(see README "Honest limitations" / architecture notes for the full story).

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return upper, lower, width


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a raw OHLCV dataframe and return it enriched with all model
    features. Rows that can't be fully computed yet (warm-up period for the
    longest rolling window) are dropped.

    Raw price-level indicators (MACD, MACD_Signal, BB_Upper, BB_Lower, ATR,
    MA_20, MA_50) are still computed and kept as intermediate columns --
    get_signal_summary() and the app's price/indicator chart use them
    directly for display -- but they are deliberately NOT part of
    FEATURE_COLS (the columns fed to the model). Only their "_pct"
    (relative-to-price) versions are model inputs, since those stay in a
    stable numeric range no matter how much a ticker's price has grown.
    """
    out = df.copy()

    out["RSI"] = _rsi(out["Close"])
    out["MACD"], out["MACD_Signal"] = _macd(out["Close"])
    out["BB_Upper"], out["BB_Lower"], out["BB_Width"] = _bollinger(out["Close"])
    out["MA_20"] = out["Close"].rolling(20).mean()
    out["MA_50"] = out["Close"].rolling(50).mean()
    out["ATR"] = _atr(out)
    out["Daily_Return"] = out["Close"].pct_change()
    out["Volatility_10d"] = out["Daily_Return"].rolling(10).std()
    out["Volume_Ratio"] = out["Volume"] / out["Volume"].rolling(20).mean().replace(0, np.nan)
    out["Price_vs_MA20"] = (out["Close"] - out["MA_20"]) / out["MA_20"].replace(0, np.nan)
    out["Price_vs_MA50"] = (out["Close"] - out["MA_50"]) / out["MA_50"].replace(0, np.nan)

    # ── Scale-invariant versions of the raw price-level indicators above ──
    # (these, not the raw ones, are what actually feed the model)
    safe_close = out["Close"].replace(0, np.nan)
    out["MACD_pct"] = out["MACD"] / safe_close
    out["MACD_Signal_pct"] = out["MACD_Signal"] / safe_close
    out["BB_Upper_pct"] = (out["BB_Upper"] - out["Close"]) / safe_close
    out["BB_Lower_pct"] = (out["BB_Lower"] - out["Close"]) / safe_close
    out["ATR_pct"] = out["ATR"] / safe_close

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Human-readable snapshot of the latest technical signals (for the UI)."""
    last = df.iloc[-1]
    signals = {}

    rsi = last["RSI"]
    signals["RSI (14)"] = f"{rsi:.1f}  " + (
        "→ Overbought" if rsi > 70 else "→ Oversold" if rsi < 30 else "→ Neutral"
    )

    macd_diff = last["MACD"] - last["MACD_Signal"]
    signals["MACD"] = "→ Bullish crossover" if macd_diff > 0 else "→ Bearish crossover"

    price = last["Close"]
    signals["Bollinger Bands"] = (
        "→ Near upper band (overbought risk)" if price >= last["BB_Upper"] * 0.98
        else "→ Near lower band (oversold risk)" if price <= last["BB_Lower"] * 1.02
        else "→ Within normal range"
    )

    signals["Trend (MA20 vs MA50)"] = (
        "→ Uptrend (MA20 > MA50)" if last["MA_20"] > last["MA_50"] else "→ Downtrend (MA20 < MA50)"
    )

    signals["Volume"] = (
        "→ Above average" if last["Volume_Ratio"] > 1.2
        else "→ Below average" if last["Volume_Ratio"] < 0.8
        else "→ Average"
    )

    return signals


# Canonical feature order fed to the model -- every entry here is a bounded
# ratio/percentage, never a raw price or volume level, so it doesn't need to
# extrapolate when a ticker's price has grown past its training-period range.
FEATURE_COLS = [
    "RSI",
    "MACD_pct", "MACD_Signal_pct",
    "BB_Upper_pct", "BB_Lower_pct", "BB_Width",
    "ATR_pct",
    "Daily_Return", "Volatility_10d",
    "Volume_Ratio",
    "Price_vs_MA20", "Price_vs_MA50",
]
