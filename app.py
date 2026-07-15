"""
app.py — Universal Market Predictor (Deluxe Edition)
=======================================================
Gradio UI serving the PRETRAINED shared model (see train.py) instead of
training a fresh LSTM on every request. Loading a cached, already-trained
model is near-instant; the old on-demand-training version took 1-2 minutes
per ticker on a free CPU Space.

If a user asks for a ticker that isn't yet in the shared model's embedding
table, the app offers a "cold-start" option: freeze the shared backbone and
fine-tune just a new embedding row for that ticker, live, in the session
(this is fast -- only one small vector is being learned, not the whole
network). That session addition is NOT written back to model_registry/
automatically (Hugging Face free Spaces storage is ephemeral anyway); to
make a new ticker permanent, run add_new_ticker.py and redeploy.

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

import os

# Must be set BEFORE `import tensorflow`. This Space may run on ZeroGPU
# hardware, which dynamically attaches/detaches a physical GPU per request --
# a model TensorFlow's CUDA context was never designed for (it assumes a
# stable device list for the life of the process, unlike PyTorch, which
# ZeroGPU is built around). Forcing TF to CPU-only sidesteps that conflict
# entirely; the shared LSTM here is small enough that CPU inference is
# already fast, so there's no real GPU speed to give up.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf

# Work around a known Keras 3.x regression: this model was saved by a Keras
# version whose layer get_config() emits a 'quantization_config' key, but the
# from_config() path used when loading (`cls(**config)`) rejects that key on
# this installed Keras version -- "Unrecognized keyword arguments passed to
# <Layer>: {'quantization_config': None}". It hits EVERY layer type that
# doesn't override from_config itself (Embedding, then Dense, then whichever
# layer comes next), because they all fall through to the same base
# implementation. Patching per-layer-type is whack-a-mole, so patch that one
# shared base class instead -- fixes every layer at once, present and future.
_layer_bases = [c for c in tf.keras.layers.Dense.__mro__ if c.__name__ == "Operation"]
_OperationBase = _layer_bases[0] if _layer_bases else None

if _OperationBase is not None:
    _orig_operation_from_config = _OperationBase.from_config.__func__

    @classmethod
    def _patched_operation_from_config(cls, config):
        if isinstance(config, dict) and "quantization_config" in config:
            config = dict(config)
            config.pop("quantization_config", None)
        return _orig_operation_from_config(cls, config)

    _OperationBase.from_config = _patched_operation_from_config

try:
    # Only present on Hugging Face Spaces configured with ZeroGPU hardware.
    # ZeroGPU refuses to start a Space unless at least one function is
    # decorated with @spaces.GPU -- it's how it knows which calls should get
    # a temporary GPU attached. Falls back to a no-op decorator everywhere
    # else (local runs, Colab, CPU-only Spaces) so this file works either way.
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(func):
        return func

sys.path.insert(0, os.path.dirname(__file__))

from src import registry
from src.data_fetcher import fetch_ohlcv
from src.indicators import add_all_indicators, get_signal_summary
from src.dataset import build_ticker_dataset, return_to_price
from src.model import freeze_backbone, unfreeze_all
from src.evaluation import evaluate_ticker
from src.charts import make_price_chart, make_backtest_chart, make_forecast_chart

REGISTRY_DIR = os.environ.get("MODEL_REGISTRY_DIR", "model_registry")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

CFG = registry.load_config(CONFIG_PATH)
META = registry.load_meta(REGISTRY_DIR)
TICKER_ID_MAP = registry.load_ticker_id_map(REGISTRY_DIR)
LOOK_BACK = META.get("look_back", CFG["model"]["look_back"])
HORIZONS = META.get("horizons", CFG["model"]["horizons"])

MODEL_PATH = os.path.join(REGISTRY_DIR, "shared_model.keras")
BASE_MODEL = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# session-only additions made via the in-app cold-start flow (id -> scaler)
SESSION_TICKERS: dict[str, dict] = {}


PRESET_STOCKS = {e["ticker"]: e for e in CFG["universe"].get("stocks", [])}
PRESET_CRYPTO = {e["ticker"]: e for e in CFG["universe"].get("crypto", [])}


# ─────────────────────────────────────────────
#  CORE PIPELINE
# ─────────────────────────────────────────────

def _resolve_scaler_and_id(ticker: str):
    if ticker in TICKER_ID_MAP:
        return TICKER_ID_MAP[ticker], registry.load_scaler(ticker, REGISTRY_DIR), False
    if ticker in SESSION_TICKERS:
        s = SESSION_TICKERS[ticker]
        return s["id"], s["scaler"], False
    return None, None, True  # unknown -> needs cold-start


def _cold_start_ticker(ticker: str, ds, progress_cb=None):
    """Freeze backbone, fine-tune a fresh embedding row for this ticker,
    in-memory only (see module docstring)."""
    global BASE_MODEL

    used_ids = set(TICKER_ID_MAP.values()) | {v["id"] for v in SESSION_TICKERS.values()}
    new_id = 0
    while new_id in used_ids:
        new_id += 1

    model = tf.keras.models.load_model(MODEL_PATH)  # fresh copy, don't mutate BASE_MODEL in place
    model = freeze_backbone(model, unfreeze_layer_names=["ticker_embedding"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["fine_tune"]["learning_rate"]),
        loss=model.loss, metrics=["mae"],
    )

    train_ids = np.full((len(ds.X_train), 1), new_id)
    test_ids = np.full((len(ds.X_test), 1), new_id)

    if progress_cb:
        progress_cb(0.5, f"Cold-starting embedding for {ticker} (session-only)...")

    model.fit(
        x=[ds.X_train, train_ids], y=ds.y_train,
        validation_data=([ds.X_test, test_ids], ds.y_test),
        epochs=CFG["fine_tune"]["epochs"],
        batch_size=min(32, max(4, len(ds.X_train) // 4)),
        verbose=0,
    )
    model = unfreeze_all(model)

    SESSION_TICKERS[ticker] = {"id": new_id, "scaler": ds.scaler, "model": model}
    return new_id, ds.scaler, model


PLACEHOLDER_PRESET = "(type custom ticker below)"


@gpu_decorator
def run_analysis(ticker_input, preset_choice, horizon_days, progress=gr.Progress()):
    preset_choice = preset_choice if preset_choice != PLACEHOLDER_PRESET else None
    ticker = (ticker_input or "").strip().upper() or preset_choice
    if not ticker:
        return None, None, None, "", "", "Please enter or select a ticker."
    if BASE_MODEL is None:
        return None, None, None, "", "", (
            "No trained shared model found in model_registry/. "
            "Run train.py (locally or via notebooks/train_on_colab.ipynb) first."
        )

    try:
        progress(0.05, desc=f"Fetching {ticker}...")
        fr = fetch_ohlcv(ticker, period=CFG["data"]["history_period"])
        if fr["error"]:
            return None, None, None, "", "", fr["error"]
        if len(fr["df"]) < CFG["data"]["min_rows_required"]:
            return None, None, None, "", "", (
                f"Only {len(fr['df'])} rows available for {ticker}; "
                f"need at least {CFG['data']['min_rows_required']}."
            )

        progress(0.15, desc="Computing technical indicators...")
        df_ind = add_all_indicators(fr["df"])

        progress(0.25, desc="Building price chart...")
        price_fig = make_price_chart(df_ind, ticker, fr["name"], fr["currency"])
        signals = get_signal_summary(df_ind)
        signals_html = "\n".join(f"{k}: {v}" for k, v in signals.items())

        ticker_id, scaler, needs_cold_start = _resolve_scaler_and_id(ticker)

        ds = build_ticker_dataset(
            df_ind, ticker, ticker_id if ticker_id is not None else 0,
            look_back=LOOK_BACK, horizons=HORIZONS,
            train_fraction=CFG["data"]["train_fraction"],
        )
        if ds is None:
            return price_fig, None, None, "", signals_html, (
                "Not enough history after indicator warm-up to build prediction windows."
            )

        if needs_cold_start:
            progress(0.40, desc=f"{ticker} is new -- cold-starting its embedding...")
            ticker_id, scaler, active_model = _cold_start_ticker(ticker, ds, progress)
            status_prefix = f"[cold-started this session, not yet permanent -- see README] "
        else:
            active_model = SESSION_TICKERS.get(ticker, {}).get("model", BASE_MODEL)
            status_prefix = ""

        progress(0.80, desc="Generating predictions...")
        test_ids = np.full((len(ds.X_test), 1), ticker_id)
        pred_returns = active_model.predict([ds.X_test, test_ids], verbose=0)

        n_features = len(ds.feature_cols)
        h_idx = HORIZONS.index(horizon_days) if horizon_days in HORIZONS else 0
        actual_h = ds.raw_close_test[:, h_idx]
        predicted_h = return_to_price(ds.anchor_close_test, pred_returns[:, h_idx])
        # ds.dates_test holds the ANCHOR ("today") date each window was made from;
        # shift by the horizon so the chart's x-axis shows the date each value is actually FOR.
        target_dates_h = ds.dates_test + pd.tseries.offsets.BDay(horizon_days)
        backtest_fig = make_backtest_chart(target_dates_h, actual_h, predicted_h, ticker, horizon_days, fr["currency"])

        # Forward-looking forecast: run the model on the single most recent window
        last_window = scaler.transform(df_ind[ds.feature_cols].values)[-LOOK_BACK:]
        last_window = last_window.reshape(1, LOOK_BACK, n_features)
        next_ids = np.array([[ticker_id]])
        forecast_returns = active_model.predict([last_window, next_ids], verbose=0)[0]
        forecast_prices = return_to_price(df_ind["Close"].iloc[-1], forecast_returns)

        recent = df_ind["Close"].tail(90)
        # business-day offsets matching each configured horizon
        forecast_dates = [recent.index[-1] + pd.tseries.offsets.BDay(h) for h in HORIZONS]
        forecast_fig = make_forecast_chart(recent.index, recent.values, forecast_dates, forecast_prices, ticker, fr["currency"])

        # -- Report card comparing model vs. honest baselines --
        report_df = evaluate_ticker(ds, df_ind, pred_returns, baselines=CFG["evaluation"]["baselines"])
        this_h_report = report_df[report_df["horizon_days"] == horizon_days]
        report_html = this_h_report.to_html(index=False, classes="metrics-table")

        curr_price = df_ind["Close"].iloc[-1]
        next_price = forecast_prices[h_idx]
        delta = next_price - curr_price
        pct = delta / curr_price * 100
        clr = "#00C896" if delta >= 0 else "#FF4B6E"
        arrow = "UP" if delta >= 0 else "DOWN"

        forecast_html = f"""
<div id="forecast-banner">
  <div style="color:#9BA3BC;font-size:.85rem;margin-bottom:6px;">
    {horizon_days}-Day Forecast &nbsp;|&nbsp; {ticker} -- {fr['name']}
  </div>
  <div class="price-big" style="color:{clr};">{next_price:,.4f} {fr['currency']}</div>
  <div style="font-size:1.1rem;margin-top:6px;color:{clr};">
    {arrow} {abs(delta):,.4f} ({pct:+.2f}%) &nbsp;vs&nbsp;
    <span style="color:#9BA3BC;">Current: {curr_price:,.4f}</span>
  </div>
  <div id="disclaimer">
    Educational/research tool only. Not financial advice. Compare the "shared_lstm" row above
    against the "naive_persistence" baseline row before trusting any of this.
  </div>
</div>
"""

        progress(1.0, desc="Done")
        status = status_prefix + f"Done: {ticker} ({fr['name']}) -- horizon {horizon_days}d"
        return price_fig, backtest_fig, forecast_fig, forecast_html, signals_html + "\n\n" + report_html, status

    except Exception as e:
        import traceback
        return None, None, None, "", "", f"Error: {e}\n\n{traceback.format_exc()}"


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0F1117 !important; color: #E0E0E0; }
.gr-box, .gr-form { background: #1E2130 !important; border-radius: 12px; }
#header-banner { background: linear-gradient(135deg, #1A1F35 0%, #12273A 50%, #1A1F35 100%);
    border: 1px solid #2F3347; border-radius: 16px; padding: 28px 36px; margin-bottom: 20px; text-align: center; }
#header-banner h1 { font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #4B9EFF, #00C896, #FFD700);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#header-banner p { color: #9BA3BC; font-size: 0.95rem; margin-top: 6px; }
#forecast-banner { background: linear-gradient(135deg, #0D2237, #12273A); border: 1px solid #4B9EFF55;
    border-radius: 12px; padding: 20px 28px; text-align: center; font-size: 1.0rem; }
#forecast-banner .price-big { font-size: 2.6rem; font-weight: 800; }
#disclaimer { background: #1A1225; border: 1px solid #FF4B6E44; border-radius: 8px;
    padding: 12px 18px; font-size: 0.82rem; color: #9BA3BC; margin-top: 12px; }
.metrics-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 10px; }
.metrics-table th, .metrics-table td { border: 1px solid #2F3347; padding: 6px 10px; text-align: right; }
.metrics-table th { background: #1A1F35; color: #9BA3BC; }
"""

HEADER_HTML = """
<div id="header-banner">
  <h1>Universal Market Predictor -- Deluxe</h1>
  <p>Shared LSTM + per-ticker embeddings &nbsp;.&nbsp; multi-horizon forecasts &nbsp;.&nbsp;
     honest baseline comparisons on every prediction</p>
  <p style="margin-top:10px;font-size:.82rem;color:#6B7394;">
    Built by <b style="color:#4B9EFF;">Ahmed Darwish</b> .
    <a href="https://github.com/eahmeddarwish" style="color:#4B9EFF;">GitHub</a>
  </p>
</div>
"""

with gr.Blocks(css=CSS, title="Universal Market Predictor - Deluxe") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Configuration")
            all_presets = ["(type custom ticker below)"] + list(PRESET_STOCKS.keys()) + list(PRESET_CRYPTO.keys())
            preset = gr.Dropdown(choices=all_presets, value=all_presets[1] if len(all_presets) > 1 else None,
                                  label="Quick Select (trained tickers)")
            ticker_box = gr.Textbox(label="Or type any ticker", placeholder="e.g. TSLA / 2222.SR / BTC-USD")
            horizon = gr.Radio(choices=HORIZONS, value=HORIZONS[0], label="Forecast Horizon (trading days)")
            analyze_btn = gr.Button("Analyze", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, value="Ready.")

            with gr.Accordion("About this model", open=False):
                gr.Markdown(
                    "One shared LSTM backbone is trained across the whole ticker universe; "
                    "each ticker gets its own learned embedding vector. New tickers typed in "
                    "that aren't yet trained get a **session-only cold-start** (fast -- only "
                    "their embedding trains, not the whole network). To make a new ticker "
                    "permanent, run `add_new_ticker.py` and redeploy.\n\n"
                    "Every prediction is shown next to a `naive_persistence` baseline "
                    "(\"tomorrow = today\") in the metrics table. If the model doesn't clearly "
                    "beat that baseline, treat the forecast with skepticism."
                )

        with gr.Column(scale=3):
            forecast_html_box = gr.HTML()
            with gr.Tabs():
                with gr.Tab("Price & Indicators"):
                    price_plot = gr.Plot()
                with gr.Tab("Multi-Horizon Forecast"):
                    forecast_plot = gr.Plot()
                with gr.Tab("Backtest (this horizon)"):
                    backtest_plot = gr.Plot()

            gr.Markdown("#### Signals & Model-vs-Baseline Report")
            signals_box = gr.HTML()

    # Preset dropdown values are already plain ticker symbols (dict keys built
    # from config.yaml), so run_analysis can use `preset` directly as the
    # fallback when the free-text ticker box is empty.
    analyze_btn.click(
        fn=run_analysis,
        inputs=[ticker_box, preset, horizon],
        outputs=[price_plot, backtest_plot, forecast_plot, forecast_html_box, signals_box, status_box],
    )

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px;color:#6B7394;font-size:.82rem;
                border-top:1px solid #2F3347;margin-top:24px;">
      Built by <b style="color:#4B9EFF;">Ahmed Darwish</b>
      <br><span style="color:#4B4F69;">Educational and research use only. Not financial advice.</span>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
