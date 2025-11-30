# nvda_trend_transforms.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Optional, Callable, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet


# ============================================================
# 1. Data download
# ============================================================

def download_price_series(
    ticker: str = "NVDA",
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily data from Yahoo Finance and keep only the Close price.
    Result: DataFrame with column 'close'.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Close"]].dropna()
    df.rename(columns={"Close": "close"}, inplace=True)
    return df


# ============================================================
# 2. Transformations on the price series
#    (all return an array of SAME length as 'prices')
# ============================================================

def _transform_close(prices: np.ndarray) -> np.ndarray:
    """Identity: y_t = close_t."""
    return prices.astype(float)


def _transform_log_close(prices: np.ndarray) -> np.ndarray:
    """Log prices: y_t = log(close_t)."""
    prices = prices.astype(float)
    return np.log(prices)


def _transform_sqrt_close(prices: np.ndarray) -> np.ndarray:
    """Square-root prices: y_t = sqrt(close_t)."""
    prices = prices.astype(float)
    return np.sqrt(prices)


def _transform_diff_close(prices: np.ndarray) -> np.ndarray:
    """
    First difference of prices: y_t = close_t - close_{t-1}.
    First element is set to NaN and later dropped.
    """
    prices = prices.astype(float)
    y = np.full_like(prices, np.nan, dtype=float)
    y[1:] = prices[1:] - prices[:-1]
    return y


def _transform_simple_return(prices: np.ndarray) -> np.ndarray:
    """
    Simple returns: y_t = close_t / close_{t-1} - 1.
    First element is NaN and later dropped.
    """
    prices = prices.astype(float)
    y = np.full_like(prices, np.nan, dtype=float)
    y[1:] = prices[1:] / prices[:-1] - 1.0
    return y


def _transform_log_return(prices: np.ndarray) -> np.ndarray:
    """
    Log returns: y_t = log(close_t) - log(close_{t-1}).
    First element is NaN and later dropped.
    """
    prices = prices.astype(float)
    logp = np.log(prices)
    y = np.full_like(logp, np.nan, dtype=float)
    y[1:] = logp[1:] - logp[:-1]
    return y


def _transform_zscore_log_close(prices: np.ndarray) -> np.ndarray:
    """
    Z-scored log prices:
      y_t = (log(close_t) - mean) / std
    """
    prices = prices.astype(float)
    logp = np.log(prices)
    mu = float(np.mean(logp))
    sigma = float(np.std(logp))
    if sigma == 0.0:
        sigma = 1.0
    return (logp - mu) / sigma


TRANSFORM_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "close": _transform_close,
    "log_close": _transform_log_close,
    "sqrt_close": _transform_sqrt_close,
    "diff_close": _transform_diff_close,
    "simple_return": _transform_simple_return,
    "log_return": _transform_log_return,
    "zscore_log_close": _transform_zscore_log_close,
}


# ============================================================
# 3. Build (X, y, dates) for a given transformation
# ============================================================

def make_transformed_time_regression_data(
    df: pd.DataFrame,
    transform_name: str,
    price_col: str = "close",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Given a DataFrame with 'close' and a transformation name, build:

      X: time index t = 0,1,...,n-1 as shape (n, 1)
      y: transformed series (same length as X)
      dates: DatetimeIndex aligned with y

    For transforms that produce NaN/inf (e.g., returns at t=0),
    those rows are dropped.
    """
    if transform_name not in TRANSFORM_FUNCS:
        raise ValueError(
            f"Unknown transform '{transform_name}'. "
            f"Available: {list(TRANSFORM_FUNCS.keys())}"
        )

    # Force 1-D float array for prices
    prices = df[price_col].to_numpy(dtype=float).ravel()

    # Apply transform and force 1-D array as well
    y_raw = TRANSFORM_FUNCS[transform_name](prices)
    y_raw = np.asarray(y_raw, dtype=float).ravel()

    if y_raw.shape[0] != prices.shape[0]:
        raise ValueError(
            f"Transform '{transform_name}' must return array of "
            f"same length as prices. Got {y_raw.shape[0]}, "
            f"expected {prices.shape[0]}."
        )

    # 1-D boolean mask
    mask = np.isfinite(y_raw)
    mask = np.asarray(mask, dtype=bool).ravel()

    # Apply mask
    y = y_raw[mask]
    # Use numpy array for dates then wrap back to DatetimeIndex
    dates_full = df.index.to_numpy()
    dates = pd.DatetimeIndex(dates_full[mask])

    # Time index
    t = np.arange(len(y), dtype=float)
    X = t.reshape(-1, 1)

    return X, y, dates



# ============================================================
# 4. Temporal split and Elastic Net model
# ============================================================

def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Split X, y into train/val/test without shuffling.
    Fractions are given for train and val; test gets the rest.
    """
    n = len(y)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_val

    if n_test <= 0:
        raise ValueError("Not enough data to allocate a test set.")

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def build_elastic_net_model(
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 0,
) -> Pipeline:
    """
    Build a pipeline: StandardScaler + ElasticNet.
    For 1D feature, scaling is not strictly necessary
    but is standard practice.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                random_state=random_state,
            )),
        ]
    )
    return model


# ============================================================
# 5. RMSE and time-weighted RMSE
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard RMSE = sqrt(mean squared error).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def weighted_rmse_linear_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Time-weighted RMSE with linearly increasing weights.
    Use scheme: w_i ∝ (i+1), normalized to sum 1.
    """
    n = len(y_true)
    idx = np.arange(n, dtype=float)
    w = (idx + 1.0) / np.sum(idx + 1.0)
    mse_w = np.sum(w * (y_true - y_pred) ** 2)
    return float(np.sqrt(mse_w))


def compute_segment_metrics(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Given a fitted model and a segment (X, y),
    return both RMSE and weighted RMSE.
    """
    y_hat = model.predict(X)
    return {
        "rmse": rmse(y, y_hat),
        "rmse_w": weighted_rmse_linear_time(y, y_hat),
    }


def _fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    random_state: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[Dict[str, Dict[str, float]], Pipeline]:
    """
    Internal helper: fit Elastic Net on train and compute metrics
    on train / val / test / train+val.
    """
    splits = temporal_split(X, y, train_frac=train_frac, val_frac=val_frac)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    model = build_elastic_net_model(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    metrics_train = compute_segment_metrics(model, X_train, y_train)
    metrics_val = compute_segment_metrics(model, X_val, y_val)
    metrics_test = compute_segment_metrics(model, X_test, y_test)

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    metrics_train_val = compute_segment_metrics(model, X_train_val, y_train_val)

    metrics = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "train_val": metrics_train_val,
    }

    return metrics, model


# ============================================================
# 6. Single-transform and multi-transform experiments
# ============================================================

def _fit_transform_on_df(
    df: pd.DataFrame,
    transform_name: str,
    alpha: float,
    l1_ratio: float,
    random_state: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[Dict[str, object], Pipeline]:
    """
    Internal helper: given df and transform_name, fit and evaluate.
    """
    X, y, _ = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )
    variance = float(np.var(y))  # population variance of transformed series

    metrics, model = _fit_and_evaluate(
        X, y,
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    result = {
        "transform": transform_name,
        "variance": variance,
        "metrics": metrics,
    }
    return result, model


def run_transform_experiment(
    transform_name: str,
    ticker: str = "NVDA",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[Dict[str, object], Pipeline, pd.DataFrame]:
    """
    Full pipeline for a SINGLE transformation:

      - Download Close prices.
      - Build transformed target y_t according to 'transform_name'.
      - Split 60/20/20 (train/val/test).
      - Fit Elastic Net on TRAIN only.
      - Compute RMSE and time-weighted RMSE for:
          train, val, test, train+val.
      - Compute variance of transformed series.

    Returns
    -------
    result : dict
        {
          'transform': str,
          'variance': float,
          'metrics': {split: {'rmse', 'rmse_w'}}
        }
    model : Pipeline
        Fitted Elastic Net pipeline.
    df : pd.DataFrame
        Original price DataFrame with 'close'.
    """
    df = download_price_series(ticker=ticker, start=start, end=end)
    result, model = _fit_transform_on_df(
        df,
        transform_name=transform_name,
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    print(f"\n=== {ticker} Elastic Net trend on transform '{transform_name}' ===")
    print(f"Alpha={alpha:.4f}, l1_ratio={l1_ratio:.2f}")
    print(f"Variance of transformed series: {result['variance']:.6e}")
    for split_name, m in result["metrics"].items():
        print(
            f"[{split_name:9s}]  "
            f"RMSE = {m['rmse']:.6f}   RMSE_w = {m['rmse_w']:.6f}"
        )

    return result, model, df


def run_all_transforms_experiment(
    transform_names: Optional[List[str]] = None,
    ticker: str = "NVDA",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Pipeline], pd.DataFrame]:
    """
    Run the experiment for MANY transformations in one shot.

    Returns
    -------
    results : dict
        results[transform_name] -> {'transform', 'variance', 'metrics'}
    models : dict
        models[transform_name] -> fitted Pipeline
    df : pd.DataFrame
        Original price DataFrame with 'close'.
    """
    if transform_names is None:
        transform_names = list(TRANSFORM_FUNCS.keys())

    df = download_price_series(ticker=ticker, start=start, end=end)

    results: Dict[str, Dict[str, object]] = {}
    models: Dict[str, Pipeline] = {}

    for name in transform_names:
        result, model = _fit_transform_on_df(
            df,
            transform_name=name,
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        results[name] = result
        models[name] = model

        print(f"\n=== {ticker} transform '{name}' ===")
        print(f"Variance: {result['variance']:.6e}")
        for split_name, m in result["metrics"].items():
            print(
                f"[{split_name:9s}]  "
                f"RMSE = {m['rmse']:.6f}   RMSE_w = {m['rmse_w']:.6f}"
            )

    return results, models, df


# Backwards-compatible wrapper for log-prices only
def run_logprice_experiment(
    ticker: str = "NVDA",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[Dict[str, Dict[str, float]], Pipeline, pd.DataFrame]:
    """
    Convenience wrapper: same as before but fixed to transform 'log_close'.
    Returns only the metrics dict (without variance) plus model and df.
    """
    result, model, df = run_transform_experiment(
        transform_name="log_close",
        ticker=ticker,
        start=start,
        end=end,
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    return result["metrics"], model, df


# ============================================================
# 7. Plotting helpers (colored train/val/test)
# ============================================================

def _segment_indices(
    n: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
):
    """
    Internal helper: compute indices for train/val/test splits.
    """
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough data to allocate a test set.")

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, None)
    return idx_train, idx_val, idx_test, n_train, n_val, n_test


def plot_transform_trend(
    df: pd.DataFrame,
    model: Pipeline,
    transform_name: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Plot the TRANSFORMED series with clear color separation of
    train / validation / test segments, plus the fitted trend line.

    Colors:
      - Train: blue
      - Validation: orange
      - Test: green

    Black dashed line = Elastic Net trend over all dates.
    """
    X_all, y_all, dates = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )
    y_hat_all = model.predict(X_all)
    n = len(y_all)

    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    plt.figure(figsize=(10, 5))

    # Data segments
    plt.plot(dates[idx_train], y_all[idx_train],
             color="tab:blue", label=f"Train ({transform_name})")
    plt.plot(dates[idx_val], y_all[idx_val],
             color="tab:orange", label=f"Val ({transform_name})")
    plt.plot(dates[idx_test], y_all[idx_test],
             color="tab:green", label=f"Test ({transform_name})")

    # Trend line across all dates
    plt.plot(dates, y_hat_all,
             color="black", linestyle="--", linewidth=2.0,
             label="Elastic Net trend")

    # Vertical lines at split points
    plt.axvline(dates[n_train], color="gray", linestyle=":", linewidth=1.0)
    plt.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=1.0)

    plt.xlabel("Date")
    plt.ylabel(f"Transformed value ({transform_name})")
    plt.title(f"NVDA: transform '{transform_name}' vs Elastic Net trend\n"
              "Train / Val / Test segments")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Specific wrappers to keep previous behavior/naming

def plot_log_trend(
    df: pd.DataFrame,
    model: Pipeline,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Wrapper for log-price plot (uses transform_name='log_close').
    """
    plot_transform_trend(
        df,
        model,
        transform_name="log_close",
        train_frac=train_frac,
        val_frac=val_frac,
    )


def plot_price_with_exp_trend(
    df: pd.DataFrame,
    model: Pipeline,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Plot Close prices with clear color separation of
    train / validation / test segments, plus exp(log-trend).

    This only makes sense for the 'log_close' transform.
    """
    # Get transformed (log_close) data and dates
    X_all, y_log_all, dates = make_transformed_time_regression_data(
        df, transform_name="log_close", price_col="close"
    )
    y_hat_log = model.predict(X_all)

    # Align raw prices to the same dates (mask the same way)
    prices_all = df["close"].to_numpy(dtype=float).ravel()
    # Recompute mask in exactly the same way as in make_transformed_time_regression_data
    logp_all = np.log(prices_all)
    mask = np.isfinite(logp_all)
    mask = np.asarray(mask, dtype=bool).ravel()

    prices = prices_all[mask]
    trend_prices = np.exp(y_hat_log)

    # Now dates, prices, trend_prices all have same length
    n = len(prices)

    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    plt.figure(figsize=(10, 5))

    # Data segments
    plt.plot(dates[idx_train], prices[idx_train],
             color="tab:blue", label="Train Close")
    plt.plot(dates[idx_val], prices[idx_val],
             color="tab:orange", label="Val Close")
    plt.plot(dates[idx_test], prices[idx_test],
             color="tab:green", label="Test Close")

    # Trend line across all dates
    plt.plot(dates, trend_prices,
             color="black", linestyle="--", linewidth=2.0,
             label="Trend (exp log)")

    # Vertical lines at split points
    plt.axvline(dates[n_train], color="gray", linestyle=":", linewidth=1.0)
    plt.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=1.0)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("NVDA: Close vs exp(Elastic Net log-trend)\nTrain / Val / Test segments")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_destransformed_price_trend(
    df: pd.DataFrame,
    model: Pipeline,
    transform_name: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Plot how the ORIGINAL price series would look after applying the
    *inverse* of the chosen transform to the fitted trend.

    - X axis: dates (subset where transform is defined).
    - Colored lines: original Close prices (train / val / test).
    - Black dashed line: "destransformed" trend in price units.

    Supported transforms:
      - 'close'            -> identity
      - 'log_close'        -> exp
      - 'sqrt_close'       -> square
      - 'zscore_log_close' -> un-zscore + exp
      - 'diff_close'       -> cumulative sum (anchored at first price)
      - 'simple_return'    -> cumulative product of (1 + r_t)
      - 'log_return'       -> cumulative sum in log-space + exp
    """
    if transform_name not in TRANSFORM_FUNCS:
        raise ValueError(
            f"Unknown transform '{transform_name}'. "
            f"Available: {list(TRANSFORM_FUNCS.keys())}"
        )

    # --- 1. Build transform and mask (independent of other functions) ---
    prices_all = df["close"].to_numpy(dtype=float).ravel()
    y_raw = TRANSFORM_FUNCS[transform_name](prices_all)
    y_raw = np.asarray(y_raw, dtype=float).ravel()

    if y_raw.shape[0] != prices_all.shape[0]:
        raise ValueError(
            f"Transform '{transform_name}' must return same length as prices "
            f"(got {y_raw.shape[0]}, expected {prices_all.shape[0]})."
        )

    mask = np.isfinite(y_raw)
    mask = np.asarray(mask, dtype=bool).ravel()

    # Subset dates and prices to positions where transform is defined
    dates_full = df.index.to_numpy()
    dates = pd.DatetimeIndex(dates_full[mask])
    prices = prices_all[mask]

    n = len(prices)
    if n == 0:
        raise ValueError(f"No finite values for transform '{transform_name}'.")

    # --- 2. Predict trend in TRANSFORM space on the same index 0..n-1 ---
    X_all = np.arange(n, dtype=float).reshape(-1, 1)
    y_hat = model.predict(X_all)

    # --- 3. "Destransform": map trend from transform space back to price ---
    if transform_name == "close":
        # y = price
        price_hat = y_hat

    elif transform_name == "log_close":
        # y = log(price)
        price_hat = np.exp(y_hat)

    elif transform_name == "sqrt_close":
        # y = sqrt(price)
        price_hat = np.clip(y_hat, a_min=0.0, a_max=None) ** 2

    elif transform_name == "zscore_log_close":
        # y = (log(price) - mu) / sigma  -> log(price) = y*sigma + mu
        logp_all = np.log(prices_all)
        mu = float(np.mean(logp_all))
        sigma = float(np.std(logp_all))
        if sigma == 0.0:
            sigma = 1.0
        log_price_hat = y_hat * sigma + mu
        price_hat = np.exp(log_price_hat)

    elif transform_name == "diff_close":
        # y_t = price_t - price_{t-1}; reconstruct via cumulative sum
        price_hat = np.empty_like(y_hat)
        price_hat[0] = prices[0]  # anchor at first actual price
        for t in range(1, n):
            price_hat[t] = price_hat[t - 1] + y_hat[t]

    elif transform_name == "simple_return":
        # y_t = price_t / price_{t-1} - 1; reconstruct multiplicatively
        price_hat = np.empty_like(y_hat)
        price_hat[0] = prices[0]
        for t in range(1, n):
            price_hat[t] = price_hat[t - 1] * (1.0 + y_hat[t])

    elif transform_name == "log_return":
        # y_t = log(price_t) - log(price_{t-1})
        log_price_hat = np.empty_like(y_hat)
        price_hat = np.empty_like(y_hat)
        log_price_hat[0] = np.log(prices[0])
        price_hat[0] = prices[0]
        for t in range(1, n):
            log_price_hat[t] = log_price_hat[t - 1] + y_hat[t]
            price_hat[t] = np.exp(log_price_hat[t])

    else:
        raise ValueError(
            f"No inverse mapping implemented for transform '{transform_name}'."
        )

    # --- 4. Color-coded train / val / test segments in PRICE space ---
    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    plt.figure(figsize=(10, 5))

    # Original price segments
    plt.plot(dates[idx_train], prices[idx_train],
             color="tab:blue", label="Train Close")
    plt.plot(dates[idx_val], prices[idx_val],
             color="tab:orange", label="Val Close")
    plt.plot(dates[idx_test], prices[idx_test],
             color="tab:green", label="Test Close")

    # Dest-transformed trend
    plt.plot(dates, price_hat,
             color="black", linestyle="--", linewidth=2.0,
             label=f"Trend implied by '{transform_name}'")

    # Vertical lines at split boundaries
    plt.axvline(dates[n_train], color="gray", linestyle=":", linewidth=1.0)
    plt.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=1.0)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(
        "Original Close vs destransformed trend\n"
        f"(transform = '{transform_name}'; train / val / test segments)"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()



# === Add these at the END of your module (py102.py / nvda_trend_transforms.py) ===

from typing import List, Dict  # if not already imported


def plot_all_transform_trends(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    transform_names: Optional[List[str]] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    For each transform in `transform_names`, plot the TRANSFORMED series
    with its Elastic Net trend, color-separated into train / val / test.

    Uses:
        plot_transform_trend(df, models[name], transform_name=name, ...)
    """
    if transform_names is None:
        transform_names = list(models.keys())

    for name in transform_names:
        print(f"\n--- Plotting transformed series for: {name} ---")
        plot_transform_trend(
            df=df,
            model=models[name],
            transform_name=name,
            train_frac=train_frac,
            val_frac=val_frac,
        )


def plot_all_destransformed_price_trends(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    transform_names: Optional[List[str]] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    For each transform in `transform_names`, plot the ORIGINAL price series
    together with the "destransformed" trend implied by that transform.

    Uses:
        plot_destransformed_price_trend(df, models[name], transform_name=name, ...)
    """
    if transform_names is None:
        transform_names = list(models.keys())

    for name in transform_names:
        print(f"\n--- Plotting destransformed price trend for: {name} ---")
        plot_destransformed_price_trend(
            df=df,
            model=models[name],
            transform_name=name,
            train_frac=train_frac,
            val_frac=val_frac,
        )


# === Add this at the END of your module (py102.py / nvda_trend_transforms.py) ===

from typing import List, Dict  # if not already imported


def subplot_all_transforms_and_destransforms(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    transform_names: Optional[List[str]] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    For each transform, make ONE figure with subplots:

        [row i, col 0]  -> transformed series y_t with Elastic Net trend
        [row i, col 1]  -> ORIGINAL Close in price units +
                           'destransformed' trend implied by that transform

    - Rows correspond to transforms in `transform_names`.
    - Col 0: transformed space.
    - Col 1: price space (after inverse transform).
    - Train / val / test segments are colored (blue / orange / green).
    - Black dashed line is the trend.

    This combines "all transforms" and "all detransformations"
    in a single subplot figure.
    """
    if transform_names is None:
        transform_names = list(models.keys())

    # Ensure a stable order
    transform_names = list(transform_names)

    n_rows = len(transform_names)
    if n_rows == 0:
        raise ValueError("No transforms/models provided.")

    # Shared Close series
    prices_all = df["close"].to_numpy(dtype=float).ravel()

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(14, 3.0 * n_rows),
        sharex=False,
    )

    # Make sure axes is 2D [row, col]
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(transform_names):
        if name not in models:
            raise KeyError(f"Model for transform '{name}' not found in models dict.")

        model = models[name]

        # === 1) TRANSFORMED SPACE ===
        ax_tr = axes[i, 0]

        X_all, y_all, dates = make_transformed_time_regression_data(
            df, transform_name=name, price_col="close"
        )
        y_hat_all = model.predict(X_all)
        n = len(y_all)

        idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
            n, train_frac=train_frac, val_frac=val_frac
        )

        # Labels only on first row for legend
        train_lbl = "Train" if i == 0 else "_nolegend_"
        val_lbl = "Val" if i == 0 else "_nolegend_"
        test_lbl = "Test" if i == 0 else "_nolegend_"
        trend_lbl = "Trend" if i == 0 else "_nolegend_"

        # Transformed series segments
        ax_tr.plot(dates[idx_train], y_all[idx_train],
                   color="tab:blue", label=train_lbl, linewidth=0.9)
        ax_tr.plot(dates[idx_val], y_all[idx_val],
                   color="tab:orange", label=val_lbl, linewidth=0.9)
        ax_tr.plot(dates[idx_test], y_all[idx_test],
                   color="tab:green", label=test_lbl, linewidth=0.9)

        # Trend in transformed space
        ax_tr.plot(dates, y_hat_all,
                   color="black", linestyle="--", linewidth=1.5, label=trend_lbl)

        # Vertical split lines
        ax_tr.axvline(dates[n_train], color="gray", linestyle=":", linewidth=0.8)
        ax_tr.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=0.8)

        ax_tr.set_ylabel(f"{name}")
        ax_tr.set_title(f"Transform: {name}")

        if i == n_rows - 1:
            ax_tr.set_xlabel("Date")

        # === 2) PRICE SPACE (DETRANSFORMED) ===
        ax_de = axes[i, 1]

        # Original prices aligned with 'dates'
        prices = df.loc[dates, "close"].to_numpy(dtype=float).ravel()
        y_hat = y_hat_all  # alias

        # Inverse mapping: transform space -> price space
        if name == "close":
            price_hat = y_hat

        elif name == "log_close":
            price_hat = np.exp(y_hat)

        elif name == "sqrt_close":
            price_hat = np.clip(y_hat, a_min=0.0, a_max=None) ** 2

        elif name == "zscore_log_close":
            # Use global mean/std of log(prices_all) to match the forward transform
            logp_all = np.log(prices_all)
            mu = float(np.mean(logp_all))
            sigma = float(np.std(logp_all))
            if sigma == 0.0:
                sigma = 1.0
            log_price_hat = y_hat * sigma + mu
            price_hat = np.exp(log_price_hat)

        elif name == "diff_close":
            price_hat = np.empty_like(y_hat)
            price_hat[0] = prices[0]
            for t in range(1, n):
                price_hat[t] = price_hat[t - 1] + y_hat[t]

        elif name == "simple_return":
            price_hat = np.empty_like(y_hat)
            price_hat[0] = prices[0]
            for t in range(1, n):
                price_hat[t] = price_hat[t - 1] * (1.0 + y_hat[t])

        elif name == "log_return":
            log_price_hat = np.empty_like(y_hat)
            price_hat = np.empty_like(y_hat)
            log_price_hat[0] = np.log(prices[0])
            price_hat[0] = prices[0]
            for t in range(1, n):
                log_price_hat[t] = log_price_hat[t - 1] + y_hat[t]
                price_hat[t] = np.exp(log_price_hat[t])

        else:
            raise ValueError(
                f"No inverse mapping implemented for transform '{name}'."
            )

        # Price-space segments
        train_lbl_p = "Train Close" if i == 0 else "_nolegend_"
        val_lbl_p = "Val Close" if i == 0 else "_nolegend_"
        test_lbl_p = "Test Close" if i == 0 else "_nolegend_"
        trend_lbl_p = "Trend (price)" if i == 0 else "_nolegend_"

        ax_de.plot(dates[idx_train], prices[idx_train],
                   color="tab:blue", label=train_lbl_p, linewidth=0.9)
        ax_de.plot(dates[idx_val], prices[idx_val],
                   color="tab:orange", label=val_lbl_p, linewidth=0.9)
        ax_de.plot(dates[idx_test], prices[idx_test],
                   color="tab:green", label=test_lbl_p, linewidth=0.9)

        ax_de.plot(dates, price_hat,
                   color="black", linestyle="--", linewidth=1.5, label=trend_lbl_p)

        ax_de.axvline(dates[n_train], color="gray", linestyle=":", linewidth=0.8)
        ax_de.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=0.8)

        ax_de.set_ylabel("Price")
        ax_de.set_title(f"Destransformed to price ({name})")
        if i == n_rows - 1:
            ax_de.set_xlabel("Date")

    # One shared legend using first row labels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles_p, labels_p = axes[0, 1].get_legend_handles_labels()
    handles_all = handles + handles_p
    labels_all = labels + labels_p

    fig.legend(
        handles_all,
        labels_all,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "NVDA transforms and implied price trends\n"
        "(left: transformed; right: detransformed to price)",
        y=1.06,
        fontsize=12,
    )

    fig.tight_layout()
    plt.show()
