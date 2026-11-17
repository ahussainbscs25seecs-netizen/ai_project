"""
features.py
===========
Converts a stock's raw OHLCV DataFrame into the discrete labels consumed by
the HMM (hmm.py) and the Bayes net (bayes.py).

This module makes NO predictions and NO decisions, and downloads NOTHING.
Data comes from data_loader.py; the caller (train.py / agent.py /
backtester.py) decides what the training window is.

INPUT
-----
DataFrame with a DatetimeIndex (oldest -> newest) and columns
Open, High, Low, Close, Volume. At least MIN_ROWS (500) rows by default.

PUBLIC FUNCTIONS
----------------
add_indicators(df)                  -> df + indicator columns (warm-up rows dropped)
fit_bins(train_df)                  -> edges (plain dict, JSON-serialisable)
make_hmm_obs(df, edges, n_symbols)  -> (obs int array, dates)   for hmm.py
make_evidence(df, edges)            -> DataFrame of labels      for bayes.py
make_forward_returns(df, horizon)   -> Series of 5-day returns  for bayes.py (mu, sigma)
make_labels(df, edges, horizon)     -> Series of "Up"/"Down"    for bayes.py training
train_index_with_embargo(...)       -> training dates minus last `horizon` days
save_edges(edges, path) / load_edges(path)

RULES THIS FILE ENFORCES
------------------------
1. Bin edges (and the Up threshold) are learned from TRAINING data only
   (fit_bins). Applying edges to later data never changes them, so no
   test-period leakage.
2. Every rolling/EWM window looks backward only. The indicator at day t
   uses days <= t.
3. Warm-up rows (where an indicator is not yet defined) are DROPPED, never
   filled with 0.
4. Dates are preserved on every output so results can be aligned to days.
5. The regime label is NOT produced here (that is hmm.py's output).
"""

import json
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Settings (move to config.py if the group prefers; defaults live here so the
# file works standalone)
# --------------------------------------------------------------------------
MIN_ROWS = 500                      # minimum clean trading days
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

VOL_WINDOW = 20                     # rolling volatility / volume average
SMA_FAST, SMA_SLOW = 5, 20
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# Return bins for HMM V1 (5 symbols): percentiles of the stock's own returns
RET5_PCTS = [10, 35, 65, 90]        # -> Big down, Down, Flat, Up, Big up
# Return and volatility bins for HMM V2 (3 x 3 = 9 symbols)
RET3_PCTS = [100 / 3, 200 / 3]
VOL3_PCTS = [100 / 3, 200 / 3]
RSI_PCTS = [100 / 3, 200 / 3]       # -> Low, Mid, High

# Volume ratio (volume / 20-day average): fixed bands from the design
VOLUME_LOW, VOLUME_HIGH = 0.8, 1.5
# Trend: relative gap (SMA5 - SMA20) / SMA20 inside +-band counts as Flat
TREND_FLAT_BAND = 0.005

# Label definition for the Bayes net target
LABEL_HORIZON = 5                   # trading days ahead
LABEL_THRESH_MULT = 0.5             # "Up" = move > 0.5 * daily_std * sqrt(horizon)
                                    # (tune 0.3-0.7 on validation)

# A daily |return| above this is almost surely a data error
# (PSX daily limit is about 7.5%)
MAX_ABS_RET = 0.20

RET5_NAMES = ["Big down", "Down", "Flat", "Up", "Big up"]
RSI_NAMES = ["Low", "Mid", "High"]


class InsufficientDataError(ValueError):
    """Raised when the DataFrame is too short or malformed."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _validate(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise InsufficientDataError("Input must be a pandas DataFrame.")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise InsufficientDataError(f"Missing columns: {missing}")
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Insufficient data: {len(df)} rows, need at least {min_rows}."
        )

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if out[REQUIRED_COLS].isna().any().any():
        raise InsufficientDataError(
            "NaN values in OHLCV. Clean them in data_loader.py (use .ffill())."
        )
    if (out["Close"] <= 0).any():
        raise InsufficientDataError("Non-positive Close prices found.")

    # Unadjusted splits look like huge one-day moves and corrupt bins/regimes
    daily = out["Close"].pct_change().abs()
    if (daily > MAX_ABS_RET).any():
        bad = daily[daily > MAX_ABS_RET].index[0].date()
        warnings.warn(
            f"Daily return above {MAX_ABS_RET:.0%} (first on {bad}). "
            "Check that data_loader uses split/dividend-adjusted prices "
            "(auto_adjust=True)."
        )
    return out


# --------------------------------------------------------------------------
# Step 1: indicators
# --------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, min_rows: int = MIN_ROWS) -> pd.DataFrame:
    """
    Add: ret, vol20, sma5, sma20, rsi, macd, macd_signal, macd_hist, vol_ratio.
    Rows where any indicator is undefined (warm-up) are dropped.
    """
    out = _validate(df, min_rows)
    close, volume = out["Close"], out["Volume"].astype(float)

    out["ret"] = close.pct_change()
    out["vol20"] = out["ret"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()

    out["sma5"] = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    out["sma20"] = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

    # RSI (Wilder smoothing)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # no losses at all -> RSI = 100
    rsi = rsi.where(~((avg_loss == 0.0) & avg_gain.notna()), 100.0)
    # no gains AND no losses (flat, illiquid stretch) -> neutral 50
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    out["rsi"] = rsi

    # MACD
    ema_fast = close.ewm(span=MACD_FAST, min_periods=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, min_periods=MACD_SLOW, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(
        span=MACD_SIGNAL, min_periods=MACD_SIGNAL, adjust=False
    ).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Volume ratio: today's volume / trailing 20-day average (includes today)
    avg_vol = volume.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    out["vol_ratio"] = volume / avg_vol.replace(0.0, np.nan)

    indicator_cols = [
        "ret", "vol20", "sma5", "sma20", "rsi",
        "macd", "macd_signal", "macd_hist", "vol_ratio",
    ]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=indicator_cols)
    if out.empty:
        raise InsufficientDataError("No rows left after dropping indicator warm-up.")
    return out


def _ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either a raw OHLCV frame or one already passed through add_indicators."""
    needed = {"ret", "vol20", "sma5", "sma20", "rsi", "macd_hist", "vol_ratio"}
    if needed.issubset(df.columns):
        return df
    # Short windows are allowed here (e.g. backtester); the 500-row rule is
    # enforced at fit_bins / training time.
    return add_indicators(df, min_rows=SMA_SLOW + MACD_SLOW + MACD_SIGNAL + 1)


# --------------------------------------------------------------------------
# Step 2: learn bin edges (TRAINING DATA ONLY)
# --------------------------------------------------------------------------
def fit_bins(train_df: pd.DataFrame, min_rows: int = MIN_ROWS) -> dict:
    """
    Learn per-stock cut points and the per-stock "Up" threshold from the
    training window.

    Returns a plain dict of floats/lists, so it can be saved as JSON.
    Pass ONLY training rows here; never the full history including the test period.
    """
    if not {"ret", "vol20", "rsi"}.issubset(train_df.columns):
        feats = add_indicators(train_df, min_rows=min_rows)
    else:
        feats = _validate_enriched(train_df, min_rows)

    ret = feats["ret"].to_numpy()
    vol = feats["vol20"].to_numpy()
    rsi = feats["rsi"].to_numpy()

    edges = {
        "ret5": _pcts(ret, RET5_PCTS),
        "ret3": _pcts(ret, RET3_PCTS),
        "vol3": _pcts(vol, VOL3_PCTS),
        "rsi3": _pcts(rsi, RSI_PCTS),
        "volume_low": VOLUME_LOW,
        "volume_high": VOLUME_HIGH,
        "trend_flat_band": TREND_FLAT_BAND,
        # per-stock "Up" threshold: half a typical 5-day swing, training data only
        "label_thresh": float(LABEL_THRESH_MULT * np.std(ret) * np.sqrt(LABEL_HORIZON)),
        "n_train_rows": int(len(feats)),
        "train_start": str(feats.index[0].date()),
        "train_end": str(feats.index[-1].date()),
    }

    # duplicate cut points -> one bin can never occur (common on illiquid stocks)
    for key in ("ret5", "ret3", "vol3", "rsi3"):
        cuts = edges[key]
        if len(set(cuts)) < len(cuts):
            warnings.warn(
                f"fit_bins: duplicate edges in '{key}' {cuts}; a bin will be empty. "
                "Make sure hmm.py smooths the emission matrix."
            )
    return edges


def _validate_enriched(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Insufficient data: {len(df)} rows, need at least {min_rows}."
        )
    out = df.dropna(subset=["ret", "vol20", "rsi"])
    if len(out) < min_rows - 60:   # allow for indicator warm-up already dropped
        raise InsufficientDataError("Too few valid indicator rows to fit bins.")
    return out


def _pcts(x: np.ndarray, pcts) -> list:
    """Percentile cut points, forced to be non-decreasing."""
    cuts = np.percentile(x, pcts)
    cuts = np.maximum.accumulate(cuts)
    return [float(c) for c in cuts]


# --------------------------------------------------------------------------
# Step 3a: HMM observations
# --------------------------------------------------------------------------
def make_hmm_obs(df: pd.DataFrame, edges: dict, n_symbols: int = 5):
    """
    Build the integer observation sequence for hmm.py.

    n_symbols = 5 : V1, return bin only (0 Big down ... 4 Big up)
    n_symbols = 9 : V2, return bin (3) * 3 + volatility bin (3)

    Returns
    -------
    obs   : np.ndarray of int, one value per trading day (oldest -> newest)
    dates : DatetimeIndex aligned with obs
    """
    feats = _ensure_indicators(df)

    if n_symbols == 5:
        obs = np.digitize(feats["ret"].to_numpy(), edges["ret5"])
    elif n_symbols == 9:
        r = np.digitize(feats["ret"].to_numpy(), edges["ret3"])
        v = np.digitize(feats["vol20"].to_numpy(), edges["vol3"])
        obs = r * 3 + v
    else:
        raise ValueError("n_symbols must be 5 (V1) or 9 (V2).")

    obs = obs.astype(int)
    assert obs.min() >= 0 and obs.max() < n_symbols
    return obs, feats.index


# --------------------------------------------------------------------------
# Step 3b: Bayes-net evidence
# --------------------------------------------------------------------------
def make_evidence(df: pd.DataFrame, edges: dict) -> pd.DataFrame:
    """
    Label DataFrame for bayes.py with columns:
      RSI    : Low / Mid / High   (stock's own percentiles)
      MACD   : Bullish / Bearish  (histogram > 0)
      Trend  : Up / Flat / Down   (SMA5 vs SMA20, +-band = Flat)
      Volume : Low / Normal / High
    Regime is NOT included; it comes from hmm.py.
    """
    feats = _ensure_indicators(df)

    rsi_idx = np.digitize(feats["rsi"].to_numpy(), edges["rsi3"])
    rsi_lab = np.array(RSI_NAMES)[rsi_idx]

    macd_lab = np.where(feats["macd_hist"].to_numpy() > 0, "Bullish", "Bearish")

    gap = (feats["sma5"] - feats["sma20"]) / feats["sma20"]
    band = edges["trend_flat_band"]
    trend_lab = np.where(gap > band, "Up", np.where(gap < -band, "Down", "Flat"))

    vr = feats["vol_ratio"].to_numpy()
    vol_lab = np.where(
        vr < edges["volume_low"], "Low",
        np.where(vr > edges["volume_high"], "High", "Normal"),
    )

    return pd.DataFrame(
        {"RSI": rsi_lab, "MACD": macd_lab, "Trend": trend_lab, "Volume": vol_lab},
        index=feats.index,
    )


# --------------------------------------------------------------------------
# Targets for Bayes-net training
# --------------------------------------------------------------------------
def _forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """horizon-day-ahead simple return; the last `horizon` rows (unknown) are dropped."""
    feats = _ensure_indicators(df)
    close = feats["Close"]
    fwd = close.shift(-horizon) / close - 1.0
    return fwd.iloc[:-horizon] if horizon > 0 else fwd


def make_forward_returns(df: pd.DataFrame, horizon: int = LABEL_HORIZON) -> pd.Series:
    """Actual horizon-day return (e.g. 0.04 = +4%). bayes.py uses it for mu and sigma."""
    fwd = _forward_return(df, horizon)
    fwd.name = "fwd_ret"
    return fwd


def make_labels(df: pd.DataFrame, edges: dict, horizon: int = LABEL_HORIZON) -> pd.Series:
    """
    "Up" if the horizon-day return exceeds this stock's own threshold
    (edges["label_thresh"], learned from training data), else "Down".
    The last `horizon` rows have no known outcome and are dropped.

    NOTE for the caller: horizon-day labels overlap, so leave a gap of `horizon`
    days between the last training row and the first validation/test row
    (see train_index_with_embargo).
    """
    fwd = _forward_return(df, horizon)
    labels = pd.Series(
        np.where(fwd > edges["label_thresh"], "Up", "Down"),
        index=fwd.index,
        name="Outcome",
    )
    return labels


def train_index_with_embargo(index: pd.DatetimeIndex, train_end, horizon: int = LABEL_HORIZON):
    """Training dates up to train_end, minus the last `horizon` days, whose labels
    look into the next period. Use this instead of slicing by hand."""
    idx = index[index <= pd.Timestamp(train_end)]
    return idx[:-horizon] if horizon > 0 else idx


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------
def save_edges(edges: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(edges, f, indent=2)


def load_edges(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Self-test on synthetic prices:  python features.py
# --------------------------------------------------------------------------
def _synthetic_prices(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.integers(500_000, 2_000_000, n)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


if __name__ == "__main__":
    raw = _synthetic_prices()
    feats = add_indicators(raw)
    train = raw.iloc[: int(len(raw) * 0.7)]

    edges = fit_bins(train)
    obs5, d5 = make_hmm_obs(raw, edges, 5)
    obs9, _ = make_hmm_obs(raw, edges, 9)
    ev = make_evidence(raw, edges)
    lab = make_labels(raw, edges)
    fwd = make_forward_returns(raw)

    print(f"raw rows={len(raw)}  after warm-up drop={len(feats)}")
    print("obs5 counts:", np.bincount(obs5, minlength=5))
    print("obs9 counts:", np.bincount(obs9, minlength=9))
    print(ev.tail(3))
    print("labels:", lab.value_counts().to_dict(), " rows:", len(lab))

    # --- checks ---
    # 1. no look-ahead: indicators on truncated data equal those on full data
    cut = raw.iloc[:700]
    f_cut, f_full = add_indicators(cut), add_indicators(raw)
    common = f_cut.index
    assert np.allclose(
        f_cut[["rsi", "macd_hist", "vol20", "vol_ratio"]].to_numpy(),
        f_full.loc[common, ["rsi", "macd_hist", "vol20", "vol_ratio"]].to_numpy(),
    ), "look-ahead detected"
    # 2. no NaNs, aligned lengths
    assert not feats.isna().any().any()
    assert len(obs5) == len(ev) == len(feats)
    # 3. labels drop the unknown tail
    assert lab.index[-1] < raw.index[-1]
    # 4. edges only from the training window
    assert edges["train_end"] <= str(raw.index[int(len(raw) * 0.7) - 1].date())
    # 5. short data is rejected
    try:
        add_indicators(raw.iloc[:100])
        raise SystemExit("short data was not rejected")
    except InsufficientDataError as e:
        print("short-data check OK:", e)
    # 6. labels and forward returns cover the same dates
    assert lab.index.equals(fwd.index)
    # 7. labels agree with the threshold
    assert ((fwd > edges["label_thresh"]).to_numpy() == (lab == "Up").to_numpy()).all()
    # 8. threshold is positive and the Up rate is reasonable (not 0% or 100%)
    assert edges["label_thresh"] > 0
    up_rate = (lab == "Up").mean()
    print(f"label_thresh={edges['label_thresh']:.4f}  Up rate={up_rate:.2%}")
    assert 0.1 < up_rate < 0.6
    # 9. embargo removes exactly `horizon` rows
    tr_end = raw.index[int(len(raw) * 0.7) - 1]
    emb = train_index_with_embargo(feats.index, tr_end)
    assert len(emb) == (feats.index <= tr_end).sum() - LABEL_HORIZON
    print("ALL CHECKS PASSED")
