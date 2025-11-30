# nvda_trend_transforms.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Optional, Callable, List, Sequence, Any
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from scipy import stats  # <--- add this if not already imported




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
    poly_degree: int = 1,
) -> Pipeline:
    """
    Regression in time with Elastic Net on a polynomial of the day index.

    poly_degree = 1  -> simple linear regression in time
    poly_degree > 1  -> polynomial regression (t, t^2, ..., t^degree)

    Pipeline:
        [optional PolynomialFeatures] -> StandardScaler -> ElasticNet
    """
    if poly_degree < 1:
        raise ValueError("poly_degree must be >= 1.")

    steps = []

    # Only add PolynomialFeatures if we actually want degree > 1
    if poly_degree > 1:
        steps.append(
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))
        )

    steps.extend([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            random_state=random_state,
        )),
    ])

    return Pipeline(steps=steps)


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
    w_i ∝ (i+1), normalized to sum 1, where i=0,...,n-1.
    More weight on later (more recent) observations.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty segment in weighted_rmse_linear_time.")
    if n == 1:
        return rmse(y_true, y_pred)

    idx = np.arange(n, dtype=float)
    w = (idx + 1.0) / np.sum(idx + 1.0)
    mse_w = np.sum(w * (y_true - y_pred) ** 2)
    return float(np.sqrt(mse_w))


def weighted_rmse_exponential_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ratio_last_first: float = 10.0,
) -> float:
    """
    Time-weighted RMSE with exponentially increasing weights in time.

    We choose weights so that:
        w_last / w_first = ratio_last_first  (default: 10x more weight
                                              on the last point).

    Implementation:
        gamma^(n-1) = ratio_last_first  ->  gamma = ratio_last_first^(1/(n-1))
        w_i ∝ gamma^i, i = 0,...,n-1, then normalized to sum 1.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty segment in weighted_rmse_exponential_time.")
    if n == 1:
        return rmse(y_true, y_pred)

    r = max(float(ratio_last_first), 1.0 + 1e-9)
    gamma = r ** (1.0 / float(n - 1))

    idx = np.arange(n, dtype=float)
    w_unnorm = gamma ** idx
    w = w_unnorm / np.sum(w_unnorm)

    mse_w = np.sum(w * (y_true - y_pred) ** 2)
    return float(np.sqrt(mse_w))


# === PATCH 1: add in py103.py (near other helpers, uses scipy.stats) =======

from scipy import stats  # already added earlier; ensure it's present


def compute_normality_stats(resid: np.ndarray) -> Dict[str, float]:
    """
    Compute normality diagnostics for a 1D array of residuals.

    Returns
    -------
    {
      'n'         : sample size,
      'mean'      : sample mean,
      'std'       : sample std (ddof=1),
      'skew'      : skewness,
      'kurt_excess': excess kurtosis (kurtosis - 3),
      'jb_stat'   : Jarque-Bera statistic,
      'jb_pvalue' : Jarque-Bera p-value
    }

    Convention: smaller jb_stat (and larger jb_pvalue) = closer to Normal.
    """
    resid = np.asarray(resid, dtype=float).ravel()
    n = resid.size
    if n < 3:
        raise ValueError("Need at least 3 residuals to compute normality stats.")

    mean = float(np.mean(resid))
    std = float(np.std(resid, ddof=1))
    skew = float(stats.skew(resid, bias=False))
    kurt_excess = float(stats.kurtosis(resid, fisher=True, bias=False))
    jb_stat, jb_p = stats.jarque_bera(resid)  # jb_stat ~ χ²_2 under Normal

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt_excess": kurt_excess,
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_p),
    }


def compute_segment_metrics(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Given a fitted model and a segment (X, y), return:
      - rmse           : unweighted RMSE
      - rmse_w_linear  : linearly weighted RMSE
      - rmse_w_exp     : exponentially weighted RMSE
    """
    y_hat = model.predict(X)
    return {
        "rmse": rmse(y, y_hat),
        "rmse_w_linear": weighted_rmse_linear_time(y, y_hat),
        "rmse_w_exp": weighted_rmse_exponential_time(y, y_hat),
    }


def _fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    random_state: int,
    train_frac: float,
    val_frac: float,
    poly_degree: int = 1,
) -> Tuple[Dict[str, Dict[str, float]], Pipeline]:
    """
    Unweighted training + evaluation for a given polynomial degree in time.
    """
    splits = temporal_split(X, y, train_frac=train_frac, val_frac=val_frac)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    model = build_elastic_net_model(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        poly_degree=poly_degree,
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




# === PATCH 1: add these helpers inside py103.py ===========================
# Place them AFTER temporal_split / build_elastic_net_model / metrics
# and BEFORE high-level run_* functions.
# ========================================================================


# === PATCH 2: helper to get VALIDATION residuals (transform + price) ======
# Put this somewhere near your other helpers in py103.py
# (e.g. close to compute_price_space_metrics_for_model).

def compute_validation_residuals_transform_and_price(
    df: pd.DataFrame,
    model: Pipeline,
    transform_name: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Compute VALIDATION residuals in both spaces:

      - Transform space (y_t vs y_hat_t)
      - Price space (Close vs price_hat_t)

    Returns a dict with:
      'dates_val'      : DatetimeIndex for validation segment
      'fitted_y_val'   : y_hat on validation
      'resid_y_val'    : y_val - y_hat_val
      'fitted_p_val'   : price_hat on validation
      'resid_p_val'    : price_val - price_hat_val
    """
    # --- 1) Transform-space data ---
    X_all, y_all, dates = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )
    y_hat_all = model.predict(X_all)
    n = len(y_all)
    if n == 0:
        raise ValueError(
            f"No finite points for transform '{transform_name}' "
            f"when computing validation residuals."
        )

    # --- 2) Price-space series & destransformation ---
    prices_all = df["close"].to_numpy(dtype=float).ravel()
    prices = df.loc[dates, "close"].to_numpy(dtype=float).ravel()

    # Destransform y_hat_all -> price_hat_all
    if transform_name == "close":
        price_hat_all = y_hat_all

    elif transform_name == "log_close":
        price_hat_all = np.exp(y_hat_all)

    elif transform_name == "sqrt_close":
        price_hat_all = np.clip(y_hat_all, a_min=0.0, a_max=None) ** 2

    elif transform_name == "zscore_log_close":
        logp_all = np.log(prices_all)
        mu = float(np.mean(logp_all))
        sigma = float(np.std(logp_all))
        if sigma == 0.0:
            sigma = 1.0
        log_price_hat = y_hat_all * sigma + mu
        price_hat_all = np.exp(log_price_hat)

    elif transform_name == "diff_close":
        price_hat_all = np.empty_like(y_hat_all)
        price_hat_all[0] = prices[0]
        for t in range(1, n):
            price_hat_all[t] = price_hat_all[t - 1] + y_hat_all[t]

    elif transform_name == "simple_return":
        price_hat_all = np.empty_like(y_hat_all)
        price_hat_all[0] = prices[0]
        for t in range(1, n):
            price_hat_all[t] = price_hat_all[t - 1] * (1.0 + y_hat_all[t])

    elif transform_name == "log_return":
        log_price_hat = np.empty_like(y_hat_all)
        price_hat_all = np.empty_like(y_hat_all)
        log_price_hat[0] = np.log(prices[0])
        price_hat_all[0] = prices[0]
        for t in range(1, n):
            log_price_hat[t] = log_price_hat[t - 1] + y_hat_all[t]
            price_hat_all[t] = np.exp(log_price_hat[t])

    else:
        raise ValueError(
            f"No price-space inverse mapping implemented for transform "
            f"'{transform_name}'."
        )

    # --- 3) Validation indices ---
    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    y_val = y_all[idx_val]
    y_hat_val = y_hat_all[idx_val]
    p_val = prices[idx_val]
    p_hat_val = price_hat_all[idx_val]
    dates_val = dates[idx_val]

    resid_y_val = y_val - y_hat_val
    resid_p_val = p_val - p_hat_val

    return {
        "dates_val": dates_val,
        "fitted_y_val": y_hat_val,
        "resid_y_val": resid_y_val,
        "fitted_p_val": p_hat_val,
        "resid_p_val": resid_p_val,
    }


# === PATCH 1: add this helper in py103.py (near the other helpers) ==========

def compute_price_space_metrics_for_model(
    df: pd.DataFrame,
    model: Pipeline,
    transform_name: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    ratio_last_first: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compute RMSE metrics in ORIGINAL PRICE SPACE for a fitted model
    on a given transform.

    For the transform 'transform_name', we:
      - Rebuild X_all, y_all, dates for that transform.
      - Predict y_hat_all.
      - Destransform y_hat_all back to price_hat.
      - Align original Close prices with 'dates'.
      - Split into train / val / test using the SAME fractions.
      - For each split, compute:
            rmse, rmse_w_linear, rmse_w_exp
        but now on (price, price_hat).
    """
    # 1) Transform space data
    X_all, y_all, dates = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )
    y_hat_all = model.predict(X_all)
    n = len(y_all)

    if n == 0:
        raise ValueError(
            f"No finite points for transform '{transform_name}' "
            f"when computing price-space metrics."
        )

    # 2) Align original Close prices
    prices_all = df["close"].to_numpy(dtype=float).ravel()
    prices = df.loc[dates, "close"].to_numpy(dtype=float).ravel()

    if len(prices) != n:
        raise RuntimeError(
            "Mismatch between length of transformed series and "
            "aligned prices in compute_price_space_metrics_for_model."
        )

    # 3) Destransform y_hat_all -> price_hat (same logic as in plots)
    if transform_name == "close":
        price_hat = y_hat_all

    elif transform_name == "log_close":
        price_hat = np.exp(y_hat_all)

    elif transform_name == "sqrt_close":
        price_hat = np.clip(y_hat_all, a_min=0.0, a_max=None) ** 2

    elif transform_name == "zscore_log_close":
        logp_all = np.log(prices_all)
        mu = float(np.mean(logp_all))
        sigma = float(np.std(logp_all))
        if sigma == 0.0:
            sigma = 1.0
        log_price_hat = y_hat_all * sigma + mu
        price_hat = np.exp(log_price_hat)

    elif transform_name == "diff_close":
        price_hat = np.empty_like(y_hat_all)
        price_hat[0] = prices[0]
        for t in range(1, n):
            price_hat[t] = price_hat[t - 1] + y_hat_all[t]

    elif transform_name == "simple_return":
        price_hat = np.empty_like(y_hat_all)
        price_hat[0] = prices[0]
        for t in range(1, n):
            price_hat[t] = price_hat[t - 1] * (1.0 + y_hat_all[t])

    elif transform_name == "log_return":
        log_price_hat = np.empty_like(y_hat_all)
        price_hat = np.empty_like(y_hat_all)
        log_price_hat[0] = np.log(prices[0])
        price_hat[0] = prices[0]
        for t in range(1, n):
            log_price_hat[t] = log_price_hat[t - 1] + y_hat_all[t]
            price_hat[t] = np.exp(log_price_hat[t])

    else:
        raise ValueError(
            f"No price-space inverse mapping implemented for transform "
            f"'{transform_name}'."
        )

    # 4) Segment indices (same as for transform-space metrics)
    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    segments = {
        "train": (prices[idx_train], price_hat[idx_train]),
        "val": (prices[idx_val], price_hat[idx_val]),
        "test": (prices[idx_test], price_hat[idx_test]),
        "train_val": (
            prices[: n_train + n_val],
            price_hat[: n_train + n_val],
        ),
    }

    metrics_price: Dict[str, Dict[str, float]] = {}
    for split_name, (p_true, p_hat) in segments.items():
        metrics_price[split_name] = {
            "rmse": rmse(p_true, p_hat),
            "rmse_w_linear": weighted_rmse_linear_time(p_true, p_hat),
            "rmse_w_exp": weighted_rmse_exponential_time(
                p_true, p_hat, ratio_last_first=ratio_last_first
            ),
        }

    return metrics_price


def _fit_and_evaluate_with_train_weight_mode(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    random_state: int,
    train_frac: float,
    val_frac: float,
    train_weight_mode: str = "plain",
    ratio_last_first: float = 10.0,
    poly_degree: int = 1,
) -> Tuple[Dict[str, Dict[str, float]], Pipeline]:
    """
    Fit Elastic Net with different TRAIN weighting schemes and evaluate.

    train_weight_mode ∈ {'plain', 'linear', 'exp'}.
    """
    train_weight_mode = train_weight_mode.lower()
    if train_weight_mode not in {"plain", "linear", "exp"}:
        raise ValueError(
            f"train_weight_mode must be one of 'plain', 'linear', 'exp'; "
            f"got {train_weight_mode!r}"
        )

    splits = temporal_split(X, y, train_frac=train_frac, val_frac=val_frac)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    n_train = len(y_train)
    if n_train == 0:
        raise ValueError("Train segment is empty; cannot fit model.")

    # Build polynomial Elastic Net
    model = build_elastic_net_model(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        poly_degree=poly_degree,
    )

    fit_kwargs = {}
    if train_weight_mode == "plain":
        # Unweighted
        pass

    elif train_weight_mode == "linear":
        idx = np.arange(n_train, dtype=float)
        w_train = (idx + 1.0) / np.sum(idx + 1.0)
        fit_kwargs["enet__sample_weight"] = w_train

    elif train_weight_mode == "exp":
        r = max(float(ratio_last_first), 1.0 + 1e-9)
        if n_train == 1:
            gamma = 1.0
        else:
            gamma = r ** (1.0 / float(n_train - 1))
        idx = np.arange(n_train, dtype=float)
        w_unnorm = gamma ** idx
        w_train = w_unnorm / np.sum(w_unnorm)
        fit_kwargs["enet__sample_weight"] = w_train

    model.fit(X_train, y_train, **fit_kwargs)

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



# === PATCH 2: normality-regularized model selection for ONE transform ======
# Put this near your other `run_*` helpers in py103.py

from itertools import product


def run_normality_regularized_model_single_transform(
    transform_name: str,
    ticker: str = "NVDA",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    # grids for hyperparameters
    alpha_grid: Sequence[float] = (0.01, 0.1, 1.0),
    l1_ratio_grid: Sequence[float] = (0.1, 0.5, 0.9),
    poly_degree_grid: Sequence[int] = (1, 2, 3),
    train_weight_modes: Sequence[str] = ("plain", "linear", "exp"),
    # train/val/test split
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 0,
    # exponential time-weight ratio (used where relevant)
    ratio_last_first: float = 10.0,
    # objective weights
    w_rmse: float = 1.0,
    w_jb: float = 1.0,
    # which space to use for normality: 'price' or 'transform'
    normality_space: str = "price",
) -> Tuple[
    Dict[str, Any],   # best_summary
    Pipeline,         # best_model
    Dict[str, Dict[str, Any]],  # all_candidates
    pd.DataFrame,
]:
    """
    Model selection for a SINGLE transform by combining:

        - Validation RMSE (in price space), and
        - Normality of validation residuals (Jarque-Bera) in chosen space.

    Objective (to MINIMIZE):

        obj = w_rmse * RMSE_val_price + w_jb * JB_stat(resid_space)

    where resid_space are residuals either in 'price' space or 'transform' space.

    Returns
    -------
    best_summary : dict with keys including
        'transform_name', 'alpha', 'l1_ratio', 'poly_degree', 'weight_mode',
        'rmse_val_price', 'jb_stat', 'jb_pvalue', 'objective', plus metrics.
    best_model : fitted Pipeline.
    all_candidates : dict keyed by a name, containing details for all combos.
    df : price DataFrame with 'close'.
    """
    normality_space = normality_space.lower()
    if normality_space not in {"price", "transform"}:
        raise ValueError("normality_space must be 'price' or 'transform'.")

    df = download_price_series(ticker=ticker, start=start, end=end)

    # Precompute transform-space data once
    X_all, y_all, _ = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )

    # Container for all candidates
    all_candidates: Dict[str, Dict[str, Any]] = {}
    best_obj = np.inf
    best_model: Optional[Pipeline] = None
    best_summary: Optional[Dict[str, Any]] = None

    combo_index = 0

    print(
        f"\n=== Normality-regularized model selection for transform "
        f"'{transform_name}' (normality_space={normality_space}) ==="
    )
    print(
        f"Grids: alpha={list(alpha_grid)}, l1_ratio={list(l1_ratio_grid)}, "
        f"degree={list(poly_degree_grid)}, weights={list(train_weight_modes)}"
    )
    print(f"Objective: obj = {w_rmse} * RMSE_val_price + {w_jb} * JB_stat\n")

    for alpha, l1_ratio, degree, wmode in product(
        alpha_grid, l1_ratio_grid, poly_degree_grid, train_weight_modes
    ):
        combo_index += 1
        combo_name = (
            f"alpha={alpha:.3g}, l1={l1_ratio:.2f}, "
            f"deg={degree}, wmode={wmode}"
        )

        # Fit model & get transform-space metrics using existing helper
        metrics_y, model = _fit_and_evaluate_with_train_weight_mode(
            X_all,
            y_all,
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            train_frac=train_frac,
            val_frac=val_frac,
            train_weight_mode=wmode,
            ratio_last_first=ratio_last_first,
            poly_degree=degree,
        )

        # Price-space metrics (for RMSE in original units)
        price_metrics = compute_price_space_metrics_for_model(
            df,
            model=model,
            transform_name=transform_name,
            train_frac=train_frac,
            val_frac=val_frac,
            ratio_last_first=ratio_last_first,
        )
        m_val_price = price_metrics["val"]

        # Validation residuals in both spaces
        res_val = compute_validation_residuals_transform_and_price(
            df,
            model=model,
            transform_name=transform_name,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        resid_y_val = res_val["resid_y_val"]
        resid_p_val = res_val["resid_p_val"]

        if normality_space == "price":
            resid_norm = resid_p_val
        else:
            resid_norm = resid_y_val

        norm_stats = compute_normality_stats(resid_norm)
        jb_stat = norm_stats["jb_stat"]

        rmse_val_price = m_val_price["rmse"]
        obj = w_rmse * rmse_val_price + w_jb * jb_stat

        candidate = {
            "transform_name": transform_name,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "poly_degree": degree,
            "weight_mode": wmode,
            "metrics_y": metrics_y,
            "metrics_price": price_metrics,
            "rmse_val_price": rmse_val_price,
            "normality_space": normality_space,
            "normality_stats": norm_stats,
            "objective": obj,
        }
        all_candidates[f"candidate_{combo_index}"] = candidate

        print(
            f"{combo_name:45s} | "
            f"VAL price RMSE={rmse_val_price:.6f} | "
            f"JB={jb_stat:.3f}, p={norm_stats['jb_pvalue']:.3f} | "
            f"obj={obj:.6f}"
        )

        if obj < best_obj:
            best_obj = obj
            best_model = model
            best_summary = candidate

    if best_model is None or best_summary is None:
        raise RuntimeError("No candidate model was selected.")

    print("\n=== BEST MODEL (smallest objective) ===")
    print(
        f"alpha={best_summary['alpha']:.4g}, "
        f"l1={best_summary['l1_ratio']:.2f}, "
        f"degree={best_summary['poly_degree']}, "
        f"weight_mode={best_summary['weight_mode']}"
    )
    print(
        f"VAL price RMSE={best_summary['rmse_val_price']:.6f}, "
        f"JB={best_summary['normality_stats']['jb_stat']:.3f}, "
        f"p={best_summary['normality_stats']['jb_pvalue']:.3f}, "
        f"objective={best_summary['objective']:.6f}"
    )

    return best_summary, best_model, all_candidates, df


# === PATCH 4: in run_weighted_training_comparison_single_transform,
#              also compare VALIDATION in price space ========================

def run_weighted_training_comparison_single_transform(
    transform_name: str,
    ticker: str = "NVDA",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    ratio_last_first: float = 10.0,
    poly_degree: int = 1,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Pipeline],
    pd.DataFrame,
]:
    """
    Compare training with different time-weightings on a SINGLE transform
    for a given polynomial degree.

    Prints both transform-space and price-space validation metrics.
    """
    df = download_price_series(ticker=ticker, start=start, end=end)
    X, y, _ = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )

    modes = ["plain", "linear", "exp"]
    results_by_mode: Dict[str, Dict[str, Dict[str, float]]] = {}
    models_by_mode: Dict[str, Pipeline] = {}

    print(
        f"\n=== Weighted-training comparison for transform '{transform_name}' "
        f"(degree={poly_degree}) ==="
    )
    print(
        f"Alpha={alpha:.4f}, l1_ratio={l1_ratio:.2f}, "
        f"ratio_last_first={ratio_last_first:.1f}"
    )

    for mode in modes:
        metrics, model = _fit_and_evaluate_with_train_weight_mode(
            X,
            y,
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            train_frac=train_frac,
            val_frac=val_frac,
            train_weight_mode=mode,
            ratio_last_first=ratio_last_first,
            poly_degree=poly_degree,
        )
        results_by_mode[mode] = metrics
        models_by_mode[mode] = model

        # Transform-space validation metrics
        m_val_y = metrics["val"]

        # Price-space validation metrics
        price_metrics = compute_price_space_metrics_for_model(
            df,
            model=model,
            transform_name=transform_name,
            train_frac=train_frac,
            val_frac=val_frac,
            ratio_last_first=ratio_last_first,
        )
        m_val_p = price_metrics["val"]

        print(
            f"[mode={mode:6s}]  "
            f"VAL y: RMSE={m_val_y['rmse']:.6f}, "
            f"lin={m_val_y['rmse_w_linear']:.6f}, "
            f"exp={m_val_y['rmse_w_exp']:.6f}   "
            f"|  VAL price: RMSE={m_val_p['rmse']:.6f}, "
            f"lin={m_val_p['rmse_w_linear']:.6f}, "
            f"exp={m_val_p['rmse_w_exp']:.6f}"
        )

    return results_by_mode, models_by_mode, df



# === PATCH 3: plotting + normality diagnostics for VALIDATION residuals ====
# Put this in py103.py (e.g. near plotting utilities).

def plot_validation_residual_diagnostics(
    df: pd.DataFrame,
    model: Pipeline,
    transform_name: str,
    space: str = "price",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Residual diagnostics on the VALIDATION set.

    Parameters
    ----------
    df : DataFrame with 'close'
    model : fitted Pipeline (with ElasticNet)
    transform_name : str
        Name of the transform used for training (e.g. 'log_close').
    space : {'transform', 'price'}
        - 'transform': residuals in y-space (y_t - y_hat_t).
        - 'price'    : residuals in Close-space (Close - price_hat_t).
    train_frac, val_frac : float
        Fractions for train and validation splits (test is the rest).

    Produces a 2x2 figure:
      (1) Histogram + normal PDF overlay
      (2) QQ-plot vs Normal
      (3) Residuals vs fitted values
      (4) Residuals vs date

    Also prints:
      - mean, std
      - t-statistic for mean=0 and p-value
      - skewness and excess kurtosis
    """
    space = space.lower()
    if space not in {"transform", "price"}:
        raise ValueError("space must be 'transform' or 'price'.")

    # --- 1) Get validation residuals ---
    res = compute_validation_residuals_transform_and_price(
        df,
        model=model,
        transform_name=transform_name,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    dates_val = res["dates_val"]

    if space == "transform":
        resid = np.asarray(res["resid_y_val"], dtype=float)
        fitted = np.asarray(res["fitted_y_val"], dtype=float)
        label_space = f"{transform_name} residuals"
    else:
        resid = np.asarray(res["resid_p_val"], dtype=float)
        fitted = np.asarray(res["fitted_p_val"], dtype=float)
        label_space = "Price residuals (Close)"

    n = len(resid)
    if n < 3:
        raise ValueError("Not enough validation points for residual diagnostics.")

    # --- 2) Basic statistics + t-test for mean=0 ---
    mean_res = float(np.mean(resid))
    std_res = float(np.std(resid, ddof=1))  # sample std
    t_stat = mean_res / (std_res / np.sqrt(n)) if std_res > 0 else np.nan
    p_val = 2 * stats.t.sf(np.abs(t_stat), df=n - 1) if std_res > 0 else np.nan

    # Skewness, excess kurtosis
    skew = float(stats.skew(resid, bias=False))
    kurt_excess = float(stats.kurtosis(resid, fisher=True, bias=False))

    print(f"\n=== Validation residual diagnostics ({space}, transform='{transform_name}') ===")
    print(f"n = {n}")
    print(f"mean(resid) = {mean_res:.6e}")
    print(f"std(resid)  = {std_res:.6e}")
    print(f"t-stat (H0: mean=0) = {t_stat:.4f},  p-value = {p_val:.4f}")
    print(f"skewness = {skew:.4f},  excess kurtosis = {kurt_excess:.4f}")

    # --- 3) Build figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_hist, ax_qq, ax_vs_fit, ax_vs_time = axes.ravel()

    # (1) Histogram + normal PDF
    ax_hist.hist(resid, bins=30, density=True, alpha=0.6, edgecolor="black")
    x_grid = np.linspace(
        mean_res - 4 * std_res,
        mean_res + 4 * std_res,
        200,
    )
    if std_res > 0:
        pdf_norm = stats.norm.pdf(x_grid, loc=mean_res, scale=std_res)
        ax_hist.plot(x_grid, pdf_norm, "r-", linewidth=1.5, label="Normal PDF")
    ax_hist.set_title("Validation residuals histogram")
    ax_hist.set_xlabel("residual")
    ax_hist.set_ylabel("density")
    ax_hist.legend()

    # (2) QQ-plot vs Normal
    stats.probplot(resid, dist="norm", plot=ax_qq)
    ax_qq.set_title("QQ-plot vs Normal (validation residuals)")

    # (3) Residuals vs fitted
    ax_vs_fit.scatter(fitted, resid, alpha=0.6, s=15)
    ax_vs_fit.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax_vs_fit.set_title("Residuals vs fitted (validation)")
    ax_vs_fit.set_xlabel("fitted values")
    ax_vs_fit.set_ylabel("residuals")

    # (4) Residuals vs time
    ax_vs_time.scatter(dates_val, resid, alpha=0.6, s=15)
    ax_vs_time.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax_vs_time.set_title("Residuals vs date (validation)")
    ax_vs_time.set_xlabel("date")
    ax_vs_time.set_ylabel("residuals")

    fig.suptitle(
        f"Validation residual diagnostics ({label_space})",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def plot_weighted_training_comparison(
    df: pd.DataFrame,
    models_by_mode: Dict[str, Pipeline],
    transform_name: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> None:
    """
    Visual comparison of the three training schemes for ONE transform.

    Figure with 2 subplots:

      - Left:  transformed series y_t, plus trend lines for each training mode.
      - Right: original Close prices, plus destransformed trend for each mode.

    Modes (expected keys of models_by_mode):
        'plain', 'linear', 'exp'
    """
    modes = ["plain", "linear", "exp"]
    colors = {
        "plain": "black",
        "linear": "tab:red",
        "exp": "tab:purple",
    }

    # ---------- Transformed series ----------
    X_all, y_all, dates = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )
    n = len(y_all)

    idx_train, idx_val, idx_test, n_train, n_val, n_test = _segment_indices(
        n, train_frac=train_frac, val_frac=val_frac
    )

    # ---------- Prepare price series aligned with this transform ----------
    prices_all = df["close"].to_numpy(dtype=float).ravel()
    y_raw = TRANSFORM_FUNCS[transform_name](prices_all)
    y_raw = np.asarray(y_raw, dtype=float).ravel()
    mask = np.isfinite(y_raw)
    prices = prices_all[mask]
    # dates is already aligned with mask via make_transformed_time_regression_data
    # so dates, y_all, prices have same length n.

    # ---------- Build figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=False)
    ax_tr, ax_pr = axes

    # === LEFT: transformed space ===
    ax_tr.plot(dates, y_all, color="lightgray", linewidth=0.8, label="y (transform)")
    for mode in modes:
        if mode not in models_by_mode:
            continue
        model = models_by_mode[mode]
        y_hat = model.predict(X_all)
        ax_tr.plot(
            dates,
            y_hat,
            color=colors.get(mode, "gray"),
            linewidth=1.5,
            label=f"trend ({mode})",
        )

    ax_tr.axvline(dates[n_train], color="gray", linestyle=":", linewidth=0.8)
    ax_tr.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=0.8)
    ax_tr.set_title(f"Transformed space ({transform_name})")
    ax_tr.set_ylabel(transform_name)
    ax_tr.set_xlabel("Date")
    ax_tr.legend()

    # === RIGHT: price space (destransformed) ===
    ax_pr.plot(dates, prices, color="lightgray", linewidth=0.8, label="Close")

    for mode in modes:
        if mode not in models_by_mode:
            continue
        model = models_by_mode[mode]
        y_hat = model.predict(X_all)

        # Inverse mapping to price, same as in subplot_all_transforms_and_destransforms
        if transform_name == "close":
            price_hat = y_hat

        elif transform_name == "log_close":
            price_hat = np.exp(y_hat)

        elif transform_name == "sqrt_close":
            price_hat = np.clip(y_hat, a_min=0.0, a_max=None) ** 2

        elif transform_name == "zscore_log_close":
            logp_all = np.log(prices_all)
            mu = float(np.mean(logp_all))
            sigma = float(np.std(logp_all))
            if sigma == 0.0:
                sigma = 1.0
            log_price_hat = y_hat * sigma + mu
            price_hat = np.exp(log_price_hat)

        elif transform_name == "diff_close":
            price_hat = np.empty_like(y_hat)
            price_hat[0] = prices[0]
            for t in range(1, n):
                price_hat[t] = price_hat[t - 1] + y_hat[t]

        elif transform_name == "simple_return":
            price_hat = np.empty_like(y_hat)
            price_hat[0] = prices[0]
            for t in range(1, n):
                price_hat[t] = price_hat[t - 1] * (1.0 + y_hat[t])

        elif transform_name == "log_return":
            log_price_hat = np.empty_like(y_hat)
            price_hat = np.empty_like(y_hat)
            log_price_hat[0] = np.log(prices[0])
            price_hat[0] = prices[0]
            for t in range(1, n):
                log_price_hat[t] = log_price_hat[t - 1] + y_hat[t]
                price_hat[t] = np.exp(log_price_hat[t])

        else:
            # If you add new transforms, extend this block accordingly.
            continue

        ax_pr.plot(
            dates,
            price_hat,
            color=colors.get(mode, "gray"),
            linewidth=1.5,
            label=f"trend ({mode})",
        )

    ax_pr.axvline(dates[n_train], color="gray", linestyle=":", linewidth=0.8)
    ax_pr.axvline(dates[n_train + n_val], color="gray", linestyle=":", linewidth=0.8)
    ax_pr.set_title(f"Price space (destransformed from {transform_name})")
    ax_pr.set_ylabel("Price")
    ax_pr.set_xlabel("Date")
    ax_pr.legend()

    fig.suptitle(
        f"Weighted training comparison for transform '{transform_name}'\n"
        "(plain vs linear vs exponential weights on TRAIN)",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


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
    poly_degree: int = 1,
) -> Tuple[Dict[str, object], Pipeline]:
    """
    Internal helper: given df and transform_name, fit and evaluate.
    """
    X, y, _ = make_transformed_time_regression_data(
        df, transform_name=transform_name, price_col="close"
    )
    variance = float(np.var(y))  # population variance of transformed series

    metrics, model = _fit_and_evaluate(
        X,
        y,
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        train_frac=train_frac,
        val_frac=val_frac,
        poly_degree=poly_degree,
    )

    result = {
        "transform": transform_name,
        "variance": variance,
        "metrics": metrics,
    }
    return result, model



# --- put this in py103.py, replacing your existing run_transform_experiment ---

# === PATCH 2: update run_transform_experiment to also print PRICE metrics ===
# Replace your existing run_transform_experiment in py103.py with this one.

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
    poly_degree: int = 1,
) -> Tuple[Dict[str, object], Pipeline, pd.DataFrame]:
    """
    Full pipeline for a SINGLE transformation.

    poly_degree controls polynomial degree in time (1 = linear, 2 = quadratic, ...).

    Prints:
      - RMSE metrics in transform space (y_t).
      - RMSE metrics in original price space (Close), using destransformed trend.
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
        poly_degree=poly_degree,
    )

    price_metrics = compute_price_space_metrics_for_model(
        df,
        model=model,
        transform_name=transform_name,
        train_frac=train_frac,
        val_frac=val_frac,
        ratio_last_first=10.0,  # same ratio as weighted_rmse_exponential_time default
    )

    print(
        f"\n=== {ticker} Elastic Net trend on transform '{transform_name}' "
        f"(degree={poly_degree}) ==="
    )
    print(f"Alpha={alpha:.4f}, l1_ratio={l1_ratio:.2f}")
    print(f"Variance of transformed series: {result['variance']:.6e}")

    print("\n--- Metrics in TRANSFORM space (y_t) and PRICE space (Close) ---")
    for split_name in ["train", "val", "test", "train_val"]:
        m_y = result["metrics"][split_name]
        m_p = price_metrics[split_name]
        print(
            f"[{split_name:9s}]  "
            f"y: RMSE={m_y['rmse']:.6f}, lin={m_y['rmse_w_linear']:.6f}, "
            f"exp={m_y['rmse_w_exp']:.6f}   "
            f"|  price: RMSE={m_p['rmse']:.6f}, lin={m_p['rmse_w_linear']:.6f}, "
            f"exp={m_p['rmse_w_exp']:.6f}"
        )

    return result, model, df




# --- in run_all_transforms_experiment, update ONLY the print block ---

# ==== PATCH FOR YOUR MODULE py103.py ======================================
# Drop this function into py103.py (replacing your existing
# run_all_transforms_experiment). It only depends on:
#   - TRANSFORM_FUNCS
#   - _fit_transform_on_df
#   - Pipeline, pd, List, Dict, Tuple, Optional
# ==========================================================================

# === PATCH 3: in run_all_transforms_experiment, also compute PRICE metrics ===
# Replace your existing run_all_transforms_experiment with this one.

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
    poly_degree: int = 1,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Pipeline], pd.DataFrame]:
    """
    Run the experiment for MANY transformations in one shot.

    For each transform, print both transform-space and price-space metrics.
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
            poly_degree=poly_degree,
        )
        results[name] = result
        models[name] = model

        price_metrics = compute_price_space_metrics_for_model(
            df,
            model=model,
            transform_name=name,
            train_frac=train_frac,
            val_frac=val_frac,
            ratio_last_first=10.0,
        )

        print(f"\n=== {ticker} transform '{name}' (degree={poly_degree}) ===")
        print(f"Variance: {result['variance']:.6e}")
        print("--- TRAIN / VAL / TEST metrics (transform vs price) ---")
        for split_name in ["train", "val", "test"]:
            m_y = result["metrics"][split_name]
            m_p = price_metrics[split_name]
            print(
                f"[{split_name:9s}]  "
                f"y: RMSE={m_y['rmse']:.6f}, lin={m_y['rmse_w_linear']:.6f}, "
                f"exp={m_y['rmse_w_exp']:.6f}   "
                f"|  price: RMSE={m_p['rmse']:.6f}, lin={m_p['rmse_w_linear']:.6f}, "
                f"exp={m_p['rmse_w_exp']:.6f}"
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
