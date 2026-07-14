"""
data_fetcher.py
===============
Universal market data fetcher — any stock exchange + any cryptocurrency,
via Yahoo Finance (yfinance). Shared by both the training pipeline and
the live Gradio app so behavior never drifts between "what we trained on"
and "what we serve".

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import time
import yfinance as yf
import pandas as pd

# Columns we standardize on everywhere downstream
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(ticker: str, period: str = "10y", retries: int = 3) -> dict:
    """
    Fetch OHLCV history for a single ticker (stock or crypto).

    Returns
    -------
    dict with keys:
        df        : pd.DataFrame indexed by date, columns = OHLCV_COLS (or None)
        name      : str  — best-effort human readable name
        currency  : str
        exchange  : str
        error     : str | None
    """
    result = {
        "df": None,
        "name": ticker,
        "currency": "USD",
        "exchange": "-",
        "error": None,
    }

    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker.upper().strip())
            df = t.history(period=period, auto_adjust=True)

            if df.empty:
                last_err = f"No data returned for '{ticker}' (period={period})."
                time.sleep(1.5)
                continue

            for col in ["Dividends", "Stock Splits", "Capital Gains"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            df = df[OHLCV_COLS].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            result["df"] = df

            try:
                info = t.info
                result["name"] = info.get("longName") or info.get("shortName") or ticker
                result["currency"] = info.get("currency", "USD")
                result["exchange"] = info.get("exchange", "-")
            except Exception:
                pass  # metadata is best-effort, never fatal

            return result

        except Exception as e:
            last_err = str(e)
            time.sleep(1.5)

    result["error"] = f"Failed to fetch '{ticker}' after {retries} attempts: {last_err}"
    return result


def fetch_universe(tickers: list[str], period: str = "10y", verbose: bool = True) -> dict:
    """
    Fetch OHLCV for a whole list of tickers.

    Returns dict: {ticker: fetch_ohlcv result}
    Tickers that fail are still included (with result["error"] set) so the
    caller can decide whether to skip them or halt.
    """
    out = {}
    for i, tk in enumerate(tickers, 1):
        if verbose:
            print(f"[{i}/{len(tickers)}] fetching {tk} ...", flush=True)
        out[tk] = fetch_ohlcv(tk, period=period)
        if verbose and out[tk]["error"]:
            print(f"    !! {out[tk]['error']}")
    return out
