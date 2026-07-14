"""
charts.py
=========
Plotly figures for the Gradio app: price + indicators, backtest
(actual vs. predicted on the held-out test set), multi-horizon forecast,
and training loss.

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK = dict(
    plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
    font=dict(color="#E0E0E0"),
)


def make_price_chart(df: pd.DataFrame, ticker: str, name: str, currency: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=("Price + Bollinger Bands + MAs", "RSI", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#00C896", decreasing_line_color="#FF4B6E",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                              line=dict(color="#4B9EFF", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                              line=dict(color="#4B9EFF", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(75,158,255,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"], name="MA20",
                              line=dict(color="#FFD700", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="MA50",
                              line=dict(color="#FF9F4B", width=1.2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                              line=dict(color="#B084FF", width=1.3)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#FF4B6E", width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#00C896", width=1, dash="dash"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                              line=dict(color="#4B9EFF", width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                              line=dict(color="#FF9F4B", width=1.3)), row=3, col=1)

    fig.update_layout(**DARK, height=700, showlegend=True,
                       title=f"{ticker} — {name} ({currency})", xaxis_rangeslider_visible=False)
    return fig


def make_backtest_chart(dates_test, actual, predicted, ticker: str, horizon_days: int, currency: str) -> go.Figure:
    """Actual vs. predicted close on the held-out test period, for one horizon."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_test, y=actual, name="Actual",
                              line=dict(color="#4B9EFF", width=2)))
    fig.add_trace(go.Scatter(x=dates_test, y=predicted, name=f"Predicted (+{horizon_days}d)",
                              line=dict(color="#FF9F4B", width=2, dash="dot")))
    fig.update_layout(**DARK, height=420,
                       title=f"{ticker} — Backtest: Actual vs Predicted ({horizon_days}-Day Horizon)",
                       yaxis_title=f"Price ({currency})")
    return fig


def make_forecast_chart(recent_dates, recent_close, forecast_dates, forecast_prices, ticker: str, currency: str) -> go.Figure:
    """Recent actual prices + a dashed continuation into the multi-horizon forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_dates, y=recent_close, name="Actual (last 90d)",
                              line=dict(color="#4B9EFF", width=2)))

    # continuous line: last actual point + forecast points
    x_fc = [recent_dates[-1]] + list(forecast_dates)
    y_fc = [recent_close[-1]] + list(forecast_prices)
    fig.add_trace(go.Scatter(x=x_fc, y=y_fc, name="Forecast", mode="lines+markers",
                              line=dict(color="#FF9F4B", width=2, dash="dash"),
                              marker=dict(size=8, symbol="diamond")))

    fig.update_layout(**DARK, height=420,
                       title=f"{ticker} — Multi-Horizon Forecast",
                       yaxis_title=f"Price ({currency})")
    return fig


def make_loss_chart(history_path=None) -> go.Figure:
    """Placeholder loss chart — the shared model is trained once offline
    (train.py), so per-request training loss isn't available at inference
    time. Kept for API symmetry / future use if per-ticker fine-tune logs
    are wired in."""
    fig = go.Figure()
    fig.update_layout(**DARK, height=300,
                       title="Training loss is logged during train.py / add_new_ticker.py runs "
                             "(see console output), not recomputed per prediction request.")
    return fig
