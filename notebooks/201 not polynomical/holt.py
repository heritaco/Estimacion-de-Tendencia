# -*- coding: utf-8 -*-
"""
holt_no_leak_sweep.py

Holt (double) exponential smoothing with STRICT train/validation separation.

Pipeline
--------
1) Download log-prices Z_t for some ticker.
2) Split into (train, validation, test).
3) For a grid of alphas in (0,1):
       - Fit Holt's linear smoothing ONLY on train.
       - Forecast validation as L_T + m * B_T, where L_T and B_T are
         the level and trend at the last training time, and m is the
         forecast horizon (1,2,...,N_val).
       - Compute J(alpha) = MSE on validation forecasts.
4) Pick alpha* = argmin J(alpha).
5) Refit on train with alpha*:
       - On train: use 1-step-ahead Holt forecasts (causal).
       - On val+test: use L_T + m * B_T (non-horizontal line with trend).
6) Plot:
       (a) J(alpha) vs alpha (minimum marked).
       (b) Log-price vs forecast (train/val/test marked).

No data leakage:
----------------
- Parameters (level and trend) are estimated only from training data.
- Validation and test are used solely for evaluating forecasts; they
  never update the smoothing states.
"""

from typing import Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================================
# 1. Data loading: log-prices
# ============================================================

def load_log_prices(
    ticker: str,
    start: str,
    end: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Download adjusted close prices from Yahoo Finance and return
    log-prices.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Integer time index 0,...,N-1.
    Z : np.ndarray, shape (N,)
        Log-prices log(Adj Close).
    meta : dict
        Metadata (ticker, dates, etc.).
    """
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError(f"No data downloaded for ticker={ticker}.")

    adj_close = data["Adj Close"].dropna()
    dates = adj_close.index
    prices = adj_close.values.astype(float)

    Z = np.log(prices)
    N = Z.size
    t = np.arange(N, dtype=int)

    meta = {
        "ticker": ticker,
        "start": dates[0].strftime("%Y-%m-%d"),
        "end": dates[-1].strftime("%Y-%m-%d"),
        "N": N,
        "dates": dates,
        "use_log": True,
    }
    return t, Z, meta


# ============================================================
# 2. Deterministic train/val/test split
# ============================================================

def split_train_val_test(
    Z: np.ndarray,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    min_train: int = 50,
    min_val: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Deterministic split into train / validation / test.

    Returns
    -------
    Z_train, Z_val, Z_test : np.ndarray
    N_train, N_val, N_test : int
    """
    Z = np.asarray(Z, dtype=float).ravel()
    N = Z.size
    if N < (min_train + min_val + 1):
        raise ValueError(
            f"Series too short (N={N}) for min_train={min_train}, "
            f"min_val={min_val}."
        )

    N_train = max(min_train, int(np.floor(frac_train * N)))
    N_val = max(min_val, int(np.floor(frac_val * N)))
    if N_train + N_val >= N:
        # Ensure at least 1 point in test
        N_train = min_train
        N_val = min_val

    N_test = N - N_train - N_val
    if N_test <= 0:
        raise ValueError("No test points left after train/val split.")

    Z_train = Z[:N_train]
    Z_val = Z[N_train:N_train + N_val]
    Z_test = Z[N_train + N_val:]

    return Z_train, Z_val, Z_test, N_train, N_val, N_test


# ============================================================
# 3. Holt (double) exponential smoothing on TRAIN ONLY
# ============================================================

def holt_linear_train(
    Z_train: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Holt's linear (double exponential) smoothing, fitted ONLY on
    training data.

    We tie the trend smoothing parameter to the level parameter:
        beta = alpha
    so there is a single hyperparameter.

    Recursion:
        level[0] = Z_train[0]
        trend[0] = Z_train[1] - Z_train[0]
        forecast[0] = Z_train[0]   (for plotting)

        for t = 1,...,N_train-1:
            forecast[t] = level[t-1] + trend[t-1]
            level[t]    = alpha * Z_train[t] + (1-alpha) * (level[t-1] + trend[t-1])
            trend[t]    = alpha * (level[t] - level[t-1]) + (1-alpha) * trend[t-1]

    Returns
    -------
    forecast_train : np.ndarray, shape (N_train,)
        1-step-ahead forecasts for training points.
    last_level : float
        level[N_train-1].
    last_trend : float
        trend[N_train-1].

    Note: validation and test are NOT used here.
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    N_train = Z_train.size
    if N_train < 2:
        raise ValueError("Need at least 2 points in training.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    level = np.zeros(N_train, dtype=float)
    trend = np.zeros(N_train, dtype=float)
    forecast = np.zeros(N_train, dtype=float)

    # Initialization
    level[0] = Z_train[0]
    trend[0] = Z_train[1] - Z_train[0]
    forecast[0] = Z_train[0]  # for plotting

    # Recursion
    for t in range(1, N_train):
        forecast[t] = level[t - 1] + trend[t - 1]
        level[t] = alpha * Z_train[t] + (1.0 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = alpha * (level[t] - level[t - 1]) + (1.0 - alpha) * trend[t - 1]

    last_level = level[-1]
    last_trend = trend[-1]
    return forecast, last_level, last_trend


# ============================================================
# 4. J(alpha) on validation (no leakage)
# ============================================================

def J_alpha_validation(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    alpha: float,
) -> float:
    """
    Validation MSE for a given alpha.

    Mechanism:
    ----------
    1) Fit Holt smoothing ONLY on Z_train -> (L_T, B_T).
    2) For each k = 0,...,N_val-1, forecast:
           m = k+1
           F_val[k] = L_T + m * B_T
    3) Compute:
           J(alpha) = mean( (Z_val[k] - F_val[k])^2 ).

    No validation point is used to update level or trend, so the
    validation error is fully out-of-sample w.r.t. the training fit.
    """
    _, last_level, last_trend = holt_linear_train(Z_train, alpha)
    N_val = Z_val.size
    if N_val == 0:
        return np.nan

    m = np.arange(1, N_val + 1, dtype=float)  # horizons 1..N_val
    F_val = last_level + m * last_trend
    err_val = Z_val - F_val
    mse_val = float(np.mean(err_val ** 2))
    return mse_val


def sweep_alpha_grid(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    alpha_grid: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate J(alpha) on a grid and return alpha* minimizing it.

    Returns
    -------
    alpha_star : float
        Argmin over the grid.
    J_vals : np.ndarray
        J(alpha) values for each grid point.
    """
    J_vals = np.zeros_like(alpha_grid, dtype=float)
    for i, a in enumerate(alpha_grid):
        J_vals[i] = J_alpha_validation(Z_train, Z_val, a)

    idx_min = int(np.nanargmin(J_vals))
    alpha_star = float(alpha_grid[idx_min])
    print(f"Best alpha* ≈ {alpha_star:.4f}, "
          f"J_min ≈ {float(J_vals[idx_min]):.6e}")
    return alpha_star, J_vals


# ============================================================
# 5. Plots
# ============================================================

def plot_J_vs_alpha(
    alpha_grid: np.ndarray,
    J_vals: np.ndarray,
    alpha_star: float,
    save_path: Optional[str] = None,
):
    """
    Plot J(alpha) vs alpha and mark the minimum.
    """
    plt.figure()
    ax = plt.gca()
    ax.plot(alpha_grid, J_vals, marker="o", linewidth=1.0)
    ax.set_xlabel(r"Suavización $s=\alpha$")
    ax.set_ylabel(r"$J(s)$: MSE en validación (Holt, train-only)")
    ax.set_title(r"Curva $J(s)$ en el conjunto de validación (sin filtrado con validación)")

    idx_min = np.nanargmin(J_vals)
    J_min = float(J_vals[idx_min])
    s_min = float(alpha_grid[idx_min])
    ax.scatter([s_min], [J_min], color="red", zorder=5,
               label=fr"Mínimo: $s^\star \approx {s_min:.3f}$")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_logprice_vs_forecast(
    t: np.ndarray,
    Z: np.ndarray,
    forecast_all: np.ndarray,
    N_train: int,
    N_val: int,
    meta: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Plot log-price vs forecast:

    - Train: 1-step-ahead Holt forecasts, fitted only on train.
    - Val+Test: linear extrapolation from last training level and trend.
    """
    Z = np.asarray(Z, dtype=float).ravel()
    forecast_all = np.asarray(forecast_all, dtype=float).ravel()
    assert Z.size == forecast_all.size == t.size

    N = Z.size
    N_test = N - N_train - N_val

    plt.figure()
    ax = plt.gca()

    ax.plot(t, Z, label=r"$\log(\text{precio})$", linewidth=1.5)
    ax.plot(t, forecast_all, label=r"Pronóstico Holt (train + extrapolación)", linewidth=1.5)

    # Vertical boundaries
    ax.axvline(N_train - 0.5, color="k", linestyle="--", linewidth=1)
    ax.axvline(N_train + N_val - 0.5, color="k", linestyle="--", linewidth=1)

    # Segment labels
    ymax = max(np.max(Z), np.max(forecast_all))
    ymin = min(np.min(Z), np.min(forecast_all))
    ytext = ymin + 0.05 * (ymax - ymin)

    ax.text(N_train / 2.0, ytext, "Train", ha="center", va="bottom")
    ax.text(N_train + N_val / 2.0, ytext, "Validation", ha="center", va="bottom")
    ax.text(N_train + N_val + N_test / 2.0, ytext, "Test", ha="center", va="bottom")

    ax.set_xlabel("Tiempo (índice)")
    ax.set_ylabel("Log-precio")
    ax.set_title(
        f"{meta['ticker']} – Holt lineal sólo entrenado en train\n"
        r"Pronóstico 1-paso en train, extrapolación lineal en validación+prueba"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


# ============================================================
# 6. Main driver
# ============================================================

def main(
    ticker: str = "NVDA",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    min_train: int = 50,
    min_val: int = 50,
    n_alpha_grid: int = 100,
    figs_dir: Optional[str] = None,
):
    """
    Run the full pipeline for one ticker.

    1) Load log-prices.
    2) Split into train / val / test.
    3) Grid-search alpha via J(alpha) on validation, using ONLY
       a model fitted on train.
    4) Refit on train with alpha* and build:
         - 1-step-ahead forecasts on train,
         - linear extrapolation L_T + m B_T on val+test.
    5) Plot J(alpha) and forecast vs original log-price.
    """
    # 1. Load data
    t, Z, meta = load_log_prices(ticker, start, end)
    print(
        f"Loaded {meta['ticker']} from {meta['start']} to {meta['end']} "
        f"(N={meta['N']}, log={meta['use_log']})."
    )

    # 2. Split
    Z_train, Z_val, Z_test, N_train, N_val, N_test = split_train_val_test(
        Z,
        frac_train=frac_train,
        frac_val=frac_val,
        min_train=min_train,
        min_val=min_val,
    )
    print(f"Split: N_train={N_train}, N_val={N_val}, N_test={N_test}")

    # 3. Alpha grid
    alpha_grid = np.linspace(0.01, 0.99, n_alpha_grid)

    # 4. Sweep J(alpha) using ONLY train fit
    alpha_star, J_vals = sweep_alpha_grid(Z_train, Z_val, alpha_grid)

    # 5. Plot J(alpha)
    if figs_dir is not None:
        import os
        os.makedirs(figs_dir, exist_ok=True)
        path_J = f"{figs_dir}/{ticker}_J_vs_alpha_holt.png"
    else:
        path_J = None
    plot_J_vs_alpha(alpha_grid, J_vals, alpha_star, save_path=path_J)

    # 6. Refit on train with alpha*, build full forecast
    forecast_train, last_level, last_trend = holt_linear_train(Z_train, alpha_star)

    # Future (validation + test) forecasts: linear extrapolation
    N_future = N_val + N_test
    m = np.arange(1, N_future + 1, dtype=float)
    forecast_future = last_level + m * last_trend

    # Concatenate: train forecasts + future forecasts
    forecast_all = np.concatenate([forecast_train, forecast_future])
    assert forecast_all.size == Z.size

    # 7. Plot log-price vs forecast
    if figs_dir is not None:
        path_ts = f"{figs_dir}/{ticker}_logprice_vs_forecast_holt.png"
    else:
        path_ts = None
    plot_logprice_vs_forecast(
        t=t,
        Z=Z,
        forecast_all=forecast_all,
        N_train=N_train,
        N_val=N_val,
        meta=meta,
        save_path=path_ts,
    )


if __name__ == "__main__":
    main()
