# nvda_log_trend.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet


# ============================================================
# 1. Data download and preprocessing
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


def make_time_regression_data(
    df: pd.DataFrame,
    price_col: str = "close",
    log_prices: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for regression.

    X: day index t = 0,1,...,T-1 as shape (T, 1).
    y: either price or log(price), depending on log_prices.
    """
    prices = df[price_col].values.astype(float)

    if log_prices:
        # Assumes strictly positive prices (true for standard equities)
        y = np.log(prices)
    else:
        y = prices

    t = np.arange(len(y), dtype=float)
    X = t.reshape(-1, 1)
    return X, y


# ============================================================
# 2. Temporal split: 60% train, 20% val, 20% test
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


# ============================================================
# 3. Elastic Net model (linear regression on time)
# ============================================================

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
# 4. RMSE and time-weighted RMSE
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard RMSE = sqrt(mean squared error).
    For log-prices, this is in 'log-price' units.
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


# ============================================================
# 5. Main experiment for log-prices
# ============================================================

def run_logprice_experiment(
    ticker: str = "NVDA",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 0,
) -> Tuple[Dict[str, Dict[str, float]], Pipeline, pd.DataFrame]:
    """
    Full pipeline on log(Close):

      - Download Close prices.
      - Build time index feature and log-price target.
      - Split 60/20/20 (train/val/test).
      - Fit Elastic Net on TRAIN only.
      - Compute RMSE and time-weighted RMSE for:
          train, val, test, train+val.

    Returns
    -------
    metrics : dict
        Nested dict metrics[split]['rmse' or 'rmse_w'].
    model : Pipeline
        Fitted Elastic Net pipeline.
    df : pd.DataFrame
        Original price DataFrame with 'close'.
    """
    # 1. Data
    df = download_price_series(ticker=ticker, start=start, end=end)
    X, y = make_time_regression_data(df, price_col="close", log_prices=True)

    # 2. Split (only for metrics, not for plotting)
    splits = temporal_split(X, y, train_frac=0.6, val_frac=0.2)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # 3. Model
    model = build_elastic_net_model(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # 4. Metrics per segment
    metrics_train = compute_segment_metrics(model, X_train, y_train)
    metrics_val = compute_segment_metrics(model, X_val, y_val)
    metrics_test = compute_segment_metrics(model, X_test, y_test)

    # 5. Train+Val combined
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    metrics_train_val = compute_segment_metrics(model, X_train_val, y_train_val)

    metrics = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "train_val": metrics_train_val,
    }

    print(f"=== {ticker} Elastic Net trend on log(Close) ===")
    print(f"Alpha={alpha:.4f}, l1_ratio={l1_ratio:.2f}")
    for split_name, m in metrics.items():
        print(
            f"[{split_name:9s}]  "
            f"RMSE = {m['rmse']:.6f}   "
            f"RMSE_w = {m['rmse_w']:.6f}"
        )

    return metrics, model, df


# ============================================================
# 6. Plotting helpers (with colored train/val/test)
# ============================================================

def _segment_indices(n: int, train_frac: float = 0.6, val_frac: float = 0.2):
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


def plot_log_trend(
    df: pd.DataFrame,
    model: Pipeline,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Plot log(close) with clear color separation of
    train / validation / test segments, plus the fitted log-trend line.

    Colors:
      - Train: blue
      - Validation: orange
      - Test: green

    Black dashed line = Elastic Net log-trend over all dates.
    """
    # Build log data and predictions
    X_all, y_all = make_time_regression_data(df, price_col="close", log_prices=True)
    y_hat_all = model.predict(X_all)
    dates = df.index
    n = len(y_all)

    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    plt.figure(figsize=(10, 5))

    # Data segments
    plt.plot(dates[idx_train], y_all[idx_train],
             color="tab:blue", label="Train log(close)")
    plt.plot(dates[idx_val], y_all[idx_val],
             color="tab:orange", label="Validation log(close)")
    plt.plot(dates[idx_test], y_all[idx_test],
             color="tab:green", label="Test log(close)")

    # Trend line (one line across all dates)
    plt.plot(dates, y_hat_all,
             color="black", linestyle="--", linewidth=2.0,
             label="Elastic Net log-trend")

    # Vertical lines at split points
    plt.axvline(dates[n_train], color="gray", linestyle=":", linewidth=1.0)
    plt.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=1.0)

    plt.xlabel("Date")
    plt.ylabel("log price")
    plt.title("NVDA: log(Close) vs Elastic Net log-trend\nTrain / Val / Test segments")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_price_with_exp_trend(
    df: pd.DataFrame,
    model: Pipeline,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Plot Close prices with clear color separation of
    train / validation / test segments, plus exp(log-trend).

    Colors:
      - Train: blue
      - Validation: orange
      - Test: green

    Black dashed line = exp(Elastic Net log-trend) over all dates.
    """
    # Log data and predictions
    X_all, y_log_all = make_time_regression_data(df, price_col="close", log_prices=True)
    y_hat_log = model.predict(X_all)

    prices = df["close"].values.astype(float)
    trend_prices = np.exp(y_hat_log)
    dates = df.index
    n = len(prices)

    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    plt.figure(figsize=(10, 5))

    # Data segments
    plt.plot(dates[idx_train], prices[idx_train],
             color="tab:blue", label="Train Close")
    plt.plot(dates[idx_val], prices[idx_val],
             color="tab:orange", label="Validation Close")
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
