"""
evaluation.py
=============
Honest evaluation for the shared model — the single most important module
in this project.

Why this exists
-----------------
A next-day stock LSTM will almost always report a low-looking MAPE, because
tomorrow's price is usually very close to today's price. That alone proves
nothing. The only way to know whether the model has learned something real
is to compare it against a "dumb" baseline that requires zero machine
learning:

    naive persistence  : predict(t+h) = price(t)              "no change"
    moving average      : predict(t+h) = mean(last N closes)   "smoothed no-change"

If the LSTM cannot beat naive persistence by a meaningful margin — on MAE,
on directional accuracy, ideally both — it is not adding value over doing
nothing, no matter how good its raw MAPE looks in isolation. Every report
this module produces puts the model's numbers directly next to the
baseline's numbers, per ticker, per horizon.

Statistical significance on directional accuracy (added after a real run)
---------------------------------------------------------------------------
A real training run on this project showed `shared_lstm` at ~52-58%
directional accuracy on several tickers, next to baselines sitting near the
50% coin-flip line. That number alone isn't proof of a real edge: with only
a few hundred test windows per ticker, a percentage a couple of points above
50 can easily be sampling noise rather than signal. So every `shared_lstm`
row now also carries a two-sided binomial test against the 50% null and a
Wilson 95% confidence interval (`direction_significance`) — the same
"don't trust a number until it's compared against a real baseline"
philosophy this module already applies to MAE, now applied to the
directional-accuracy claim itself. `Direction_Significant` is only True when
the 95% CI excludes 50% entirely. This is intentionally NOT computed inside
`walk_forward_folds` — each fold's sample is too small for the test to be
meaningful there; it belongs on the full-test-period comparison report.

R2_vs_naive
------------
A model can report R2 = 0.95 on next-day price and that can still be almost
entirely explained by ordinary price autocorrelation (tomorrow's price
sits close to today's, so almost any reasonable prediction scores a high
R2) rather than genuine forecasting skill. Every row in the comparison
report now also carries `R2_vs_naive` — that method's R2 minus
naive_persistence's R2 for the exact same ticker/horizon — so the real lift
(or lack of it) over doing nothing is visible directly, without the reader
having to hunt down the baseline row and subtract by hand.

Walk-forward evaluation
-------------------------
Rather than a single train/test split producing one aggregate score, the
test period is chopped into K sequential folds (chronological, no shuffling)
so we can see whether performance is stable over time or whether the model
quietly falls apart in a particular regime (e.g. a volatility spike). This
does NOT retrain the model per fold — full re-fit per fold, per ticker,
across the whole universe would be far too slow to run repeatedly on a
free Colab GPU. It evaluates the one trained model's error across
consecutive chronological slices of the held-out test set. That is a
real, honest limitation and is documented as such in the README rather than
being quietly glossed over.

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import binomtest, norm

from src.dataset import TickerDataset, return_to_price


# ─────────────────────────────────────────────
#  BASELINES
# ─────────────────────────────────────────────

def naive_persistence_predict(ds: TickerDataset) -> np.ndarray:
    """predict(t+h) = price(t) for every horizon — the 'no change' baseline."""
    n_h = len(ds.horizons)
    return np.tile(ds.anchor_close_test.reshape(-1, 1), (1, n_h))


def moving_average_predict(ds: TickerDataset, df_ind: pd.DataFrame, window: int = 5) -> np.ndarray:
    """
    predict(t+h) = mean of the `window` closes ending on the anchor day,
    for every horizon (a smoothed version of 'no change').
    """
    close = df_ind["Close"]
    ma = close.rolling(window).mean()
    ma_at_anchor = ma.reindex(ds.dates_test).values
    n_h = len(ds.horizons)
    return np.tile(ma_at_anchor.reshape(-1, 1), (1, n_h))


BASELINE_FUNCS = {
    "naive_persistence": lambda ds, df_ind: naive_persistence_predict(ds),
    "moving_average_5": lambda ds, df_ind: moving_average_predict(ds, df_ind, window=5),
    "moving_average_20": lambda ds, df_ind: moving_average_predict(ds, df_ind, window=20),
}


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE / R2 / directional accuracy for one horizon's worth
    of (true, predicted) price arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE (%)": round(float(mape), 2),
        "R2": round(float(r2), 4),
        "n": int(len(y_true)),
    }


def direction_hits(anchor_price: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    """
    Raw (n_correct, n_total) counts behind the directional-accuracy
    percentage -- kept separate from `directional_accuracy()` because the
    significance test needs the raw counts, not just the rounded percent.
    """
    true_dir = np.sign(np.asarray(y_true) - np.asarray(anchor_price))
    pred_dir = np.sign(np.asarray(y_pred) - np.asarray(anchor_price))
    matches = true_dir == pred_dir
    return int(matches.sum()), int(len(matches))


def directional_accuracy(anchor_price: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    % of cases where the predicted direction of change (vs. the anchor/today
    price) matches the actual direction of change. 50% = coin flip.
    """
    n_correct, n_total = direction_hits(anchor_price, y_true, y_pred)
    return round(float(n_correct / n_total * 100), 1) if n_total else float("nan")


def _wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion, returned as
    (low, high) in percent. Preferred over the naive normal-approximation
    interval because it stays well-behaved even when the proportion is near
    50% with a moderate sample size (which is exactly this project's case).
    """
    if n_total == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = n_correct / n_total
    denom = 1 + z ** 2 / n_total
    center = phat + z ** 2 / (2 * n_total)
    margin = z * np.sqrt(phat * (1 - phat) / n_total + z ** 2 / (4 * n_total ** 2))
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (round(low * 100, 1), round(high * 100, 1))


def direction_significance(n_correct: int, n_total: int) -> dict:
    """
    Is this ticker/horizon's directional accuracy actually distinguishable
    from a 50% coin flip, or could it just be noise on a few hundred test
    windows? Two-sided exact binomial test against p=0.5, plus a Wilson 95%
    CI. `significant_95` is True only when that CI excludes 50% entirely --
    i.e. the edge (or deficit) survives an honest statistical check, not
    just eyeballing a percentage above or below 50.
    """
    if n_total == 0:
        return {"p_value": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "significant_95": False}
    result = binomtest(n_correct, n_total, p=0.5, alternative="two-sided")
    ci_low, ci_high = _wilson_ci(n_correct, n_total)
    significant = (ci_low > 50.0) or (ci_high < 50.0)
    return {
        "p_value": round(float(result.pvalue), 4),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant_95": bool(significant),
    }


# ─────────────────────────────────────────────
#  MODEL vs BASELINE REPORT (per ticker, per horizon)
# ─────────────────────────────────────────────

def evaluate_ticker(
    ds: TickerDataset,
    df_ind: pd.DataFrame,
    model_pred_returns: np.ndarray,   # model's raw predicted % return per horizon, shape (n, n_horizons)
    baselines: list[str] = ("naive_persistence", "moving_average_5"),
) -> pd.DataFrame:
    """
    Build a per-horizon comparison table: model vs each requested baseline.
    Returns a tidy DataFrame with one row per (horizon, method).

    Two additions beyond the original metrics:
      - R2_vs_naive: every method's R2 minus naive_persistence's R2 for that
        exact ticker/horizon (see module docstring).
      - Direction_CI_95 / Direction_p_value / Direction_Significant: only
        populated for the shared_lstm row -- baselines are the null model
        being tested against, not something that itself needs a
        significance claim.
    """
    rows = []

    for h_idx, h in enumerate(ds.horizons):
        y_true_h = ds.raw_close_test[:, h_idx]
        anchor = ds.anchor_close_test
        horizon_rows = []

        model_pred_h = return_to_price(anchor, model_pred_returns[:, h_idx])
        m = compute_metrics(y_true_h, model_pred_h)
        n_correct, n_total = direction_hits(anchor, y_true_h, model_pred_h)
        m["Direction Acc (%)"] = round(n_correct / n_total * 100, 1) if n_total else float("nan")
        sig = direction_significance(n_correct, n_total)
        m["Direction_p_value"] = sig["p_value"]
        m["Direction_CI_95"] = f"{sig['ci_low']}–{sig['ci_high']}%"
        m["Direction_Significant"] = sig["significant_95"]
        m.update({"ticker": ds.ticker, "horizon_days": h, "method": "shared_lstm"})
        horizon_rows.append(m)

        for name in baselines:
            pred_full = BASELINE_FUNCS[name](ds, df_ind)
            pred_h = pred_full[:, h_idx]
            bm = compute_metrics(y_true_h, pred_h)
            bm["Direction Acc (%)"] = directional_accuracy(anchor, y_true_h, pred_h)
            bm["Direction_p_value"] = float("nan")
            bm["Direction_CI_95"] = "—"
            bm["Direction_Significant"] = False
            bm.update({"ticker": ds.ticker, "horizon_days": h, "method": name})
            horizon_rows.append(bm)

        # R2 lift over doing-nothing, computed once per horizon and applied
        # to every method's row for that horizon (see module docstring).
        naive_r2 = next((r["R2"] for r in horizon_rows if r["method"] == "naive_persistence"), None)
        for r in horizon_rows:
            r["R2_vs_naive"] = round(r["R2"] - naive_r2, 4) if naive_r2 is not None else float("nan")

        rows.extend(horizon_rows)

    return pd.DataFrame(rows)[
        ["ticker", "horizon_days", "method", "MAE", "RMSE", "MAPE (%)", "R2", "R2_vs_naive",
         "Direction Acc (%)", "Direction_CI_95", "Direction_p_value", "Direction_Significant", "n"]
    ]


# ─────────────────────────────────────────────
#  WALK-FORWARD (chronological folds over the held-out test period)
# ─────────────────────────────────────────────

def walk_forward_folds(
    ds: TickerDataset,
    df_ind: pd.DataFrame,
    model_pred_returns: np.ndarray,
    n_folds: int = 5,
    horizon_idx: int = 0,
) -> pd.DataFrame:
    """
    Split the test period into `n_folds` chronological chunks and report
    MAE / Direction Accuracy per chunk, for one horizon (default: the
    shortest one). Reveals whether the model degrades in specific periods
    rather than hiding that behind a single averaged number.

    Deliberately does NOT run the significance test from `evaluate_ticker`
    here -- each fold only has a fraction of the already-modest full test
    set, so a per-fold binomial test would mostly just report "not enough
    data to tell," which isn't useful. Significance belongs on the
    full-period comparison report.
    """
    y_true = ds.raw_close_test[:, horizon_idx]
    anchor = ds.anchor_close_test
    model_pred = return_to_price(anchor, model_pred_returns[:, horizon_idx])

    fold_bounds = np.array_split(np.arange(len(y_true)), n_folds)
    rows = []
    for fold_i, idxs in enumerate(fold_bounds, 1):
        if len(idxs) == 0:
            continue
        m = compute_metrics(y_true[idxs], model_pred[idxs])
        m["Direction Acc (%)"] = directional_accuracy(anchor[idxs], y_true[idxs], model_pred[idxs])
        m.update({
            "ticker": ds.ticker,
            "fold": fold_i,
            "start_date": str(ds.dates_test[idxs[0]].date()),
            "end_date": str(ds.dates_test[idxs[-1]].date()),
        })
        rows.append(m)

    return pd.DataFrame(rows)[
        ["ticker", "fold", "start_date", "end_date", "MAE", "MAPE (%)", "Direction Acc (%)", "n"]
    ]
