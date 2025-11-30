# -*- coding: utf-8 -*-
"""
holt_winters_grid_alpha_season.py

Holt–Winters additive (triple exponential smoothing) with STRICT
train/validation separation and a 2D grid-search over:

    - alpha (smoothness) in (0, 1),
    - season_length m in a discrete set (e.g. [5, 10, 20]).

Pipeline
--------
1) Download log-prices Z_t for some ticker.
2) Split into (train, validation, test).
3) For each season_length m in season_grid and alpha in alpha_grid:
       - Fit Holt–Winters additive ONLY on train.
       - Forecast validation multi-step ahead using last training
         level/trend/season.
       - Compute J(alpha, m) = MSE on validation forecasts.
4) Pick (alpha*, m*) = argmin J(alpha, m).
5) Refit on train with (alpha*, m*):
       - On train: fitted HW values (smoothed).
       - On val+test: multi-step forecasts from end of train.
6) Plot:
       (a) Heatmap of J(alpha, m) over grid.
       (b) Log-price vs forecast (train/val/test marked).

No data leakage:
----------------
- Level, trend, and seasonal indices are updated ONLY on train.
- Validation and test never update states; they are only used
  to measure forecast accuracy.

Dependencies: numpy, matplotlib, yfinance
"""

from typing import Tuple, Dict, Any, Optional, Sequence

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

    adj_close = data["Close"].dropna()
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
# 3. Holt–Winters additive on TRAIN ONLY
# ============================================================

def _initialize_hw_additive(
    Z_train: np.ndarray,
    season_length: int,
) -> Tuple[float, float, np.ndarray]:
    """
    Initialize level, trend, and seasonal components for additive
    Holt–Winters from training data.

    Uses at least the first two full seasons if available.

    Returns
    -------
    l0 : float
        Initial level.
    b0 : float
        Initial trend (per time step).
    s0 : np.ndarray, shape (m,)
        Initial seasonal indices (additive), one per position.
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    N_train = Z_train.size
    m = season_length
    if N_train < 2 * m:
        raise ValueError(
            f"Need at least 2*season_length points in train; "
            f"N_train={N_train}, season_length={m}."
        )

    n_seasons = N_train // m  # number of complete seasons
    # Use at least 2 complete seasons, up to some cap (e.g., 8)
    n_use = max(2, min(n_seasons, 8))

    # Seasonal means for each season
    season_means = []
    for k in range(n_use):
        start = k * m
        end = start + m
        season_means.append(Z_train[start:end].mean())
    season_means = np.asarray(season_means)

    # Overall mean for those seasons
    overall_mean = season_means.mean()

    # Initial trend from first two seasons (mean difference / m)
    if n_use >= 2:
        b0 = (season_means[1] - season_means[0]) / m
    else:
        b0 = 0.0

    # Initial seasonal indices (additive deviations)
    s0 = np.zeros(m, dtype=float)
    for j in range(m):
        vals = []
        for k in range(n_use):
            idx = k * m + j
            vals.append(Z_train[idx] - season_means[k])
        s0[j] = np.mean(vals)

    l0 = overall_mean
    return l0, b0, s0


def holt_winters_additive_train(
    Z_train: np.ndarray,
    alpha: float,
    season_length: int,
    gamma: Optional[float] = None,
) -> Tuple[np.ndarray, float, float, np.ndarray, int]:
    """
    Holt–Winters additive (triple exponential) smoothing on TRAIN ONLY.

    We use:
        beta = alpha  (trend smoothing)
        gamma = gamma if provided else alpha (seasonal smoothing)

    Components:
        level_t, trend_t, seasonal[pos] with pos = t mod m.

    Recursion (component form, additive):
        l_t = alpha * (y_t - s_{pos}) + (1-alpha) * (l_{t-1} + b_{t-1})
        b_t = alpha * (l_t - l_{t-1}) + (1-alpha) * b_{t-1}
        s_{pos} = gamma * (y_t - l_t) + (1-gamma) * s_{pos}

    Fitted value at time t for plotting:
        y_hat_t = l_{t-1} + b_{t-1} + s_{pos}
    (for t = 0 we set y_hat_0 = y_0).

    Parameters
    ----------
    Z_train : np.ndarray
        Training series.
    alpha : float in (0, 1)
        Level/trend smoothing parameter.
    season_length : int
        Seasonal period m.
    gamma : float or None
        Seasonal smoothing parameter. If None, gamma = alpha.

    Returns
    -------
    fitted_train : np.ndarray, shape (N_train,)
        In-sample fitted values on train (smoothed).
    last_level : float
        Level at the last training time.
    last_trend : float
        Trend at the last training time.
    seasonal : np.ndarray, shape (m,)
        Seasonal indices at the end of training.
    last_pos : int
        Season position of the last training observation.
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    N_train = Z_train.size
    if N_train < 2:
        raise ValueError("Need at least 2 points in training.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    m = int(season_length)
    if m <= 0:
        raise ValueError("season_length must be positive.")

    if gamma is None:
        gamma = alpha
    beta = alpha

    # Initialize components
    l, b, s = _initialize_hw_additive(Z_train, m)

    fitted = np.zeros(N_train, dtype=float)

    for t in range(N_train):
        y_t = Z_train[t]
        pos = t % m

        # 1-step-ahead fitted value using previous (l, b, s[pos])
        if t == 0:
            fitted[t] = y_t  # no prior info for first point
        else:
            fitted[t] = l + b + s[pos]

        # Update components with actual observation y_t
        new_l = alpha * (y_t - s[pos]) + (1.0 - alpha) * (l + b)
        new_b = beta * (new_l - l) + (1.0 - beta) * b
        new_s = gamma * (y_t - new_l) + (1.0 - gamma) * s[pos]

        l, b = new_l, new_b
        s[pos] = new_s

    last_pos = (N_train - 1) % m
    return fitted, l, b, s, last_pos


# ============================================================
# 4. J(alpha, m) on validation (no leakage)
# ============================================================

def J_alpha_m_validation(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    alpha: float,
    season_length: int,
    gamma: Optional[float] = None,
) -> float:
    """
    Validation MSE for given (alpha, season_length) in Holt–Winters.

    Mechanism:
    ----------
    1) Fit Holt–Winters ONLY on Z_train -> (L_T, B_T, s_vec, last_pos).
    2) For each k = 0,...,N_val-1 (horizon h = k+1):
           pos_future = (last_pos + h) % m
           F_val[k] = L_T + h * B_T + s_vec[pos_future]
    3) Compute:
           J(alpha, m) = mean( (Z_val[k] - F_val[k])^2 ).

    No validation point is used to update level, trend, or season.
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()
    N_train = Z_train.size
    m = int(season_length)

    # If not enough data for that season length, return NaN (skip)
    if N_train < 2 * m:
        return np.nan

    try:
        _, last_level, last_trend, seasonal, last_pos = (
            holt_winters_additive_train(Z_train, alpha, m, gamma=gamma)
        )
    except ValueError:
        return np.nan

    N_val = Z_val.size
    if N_val == 0:
        return np.nan

    F_val = np.zeros(N_val, dtype=float)
    for k in range(N_val):
        h = k + 1  # horizon from last training point
        pos_future = (last_pos + h) % m
        F_val[k] = last_level + h * last_trend + seasonal[pos_future]

    err_val = Z_val - F_val
    mse_val = float(np.mean(err_val ** 2))
    return mse_val


def sweep_alpha_season_grid(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    alpha_grid: np.ndarray,
    season_grid: Sequence[int],
    gamma: Optional[float] = None,
) -> Tuple[float, int, np.ndarray]:
    """
    Evaluate J(alpha, m) on a 2D grid and return (alpha*, m*)
    minimizing validation MSE.

    Parameters
    ----------
    alpha_grid : np.ndarray
        Candidate alphas in (0, 1).
    season_grid : sequence of int
        Candidate season lengths m.
    gamma : float or None
        Seasonal smoothing parameter (shared across grid). If None,
        gamma = alpha (so gamma varies with alpha).

    Returns
    -------
    alpha_star : float
    season_star : int
    J_mat : np.ndarray, shape (len(season_grid), len(alpha_grid))
        Validation MSE for each (m_i, alpha_j).
    """
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    season_grid = list(season_grid)

    J_mat = np.full((len(season_grid), len(alpha_grid)), np.nan, dtype=float)

    for i, m in enumerate(season_grid):
        for j, a in enumerate(alpha_grid):
            if gamma is None:
                gamma_ij = None  # => gamma = alpha inside
            else:
                gamma_ij = gamma
            J_mat[i, j] = J_alpha_m_validation(
                Z_train, Z_val, alpha=a, season_length=m, gamma=gamma_ij
            )

    if np.all(np.isnan(J_mat)):
        raise RuntimeError("All J(alpha, m) are NaN; check grid and data.")

    idx_flat = np.nanargmin(J_mat)
    i_best, j_best = np.unravel_index(idx_flat, J_mat.shape)
    season_star = season_grid[i_best]
    alpha_star = float(alpha_grid[j_best])

    print(
        f"Best (alpha*, m*): alpha* ≈ {alpha_star:.4f}, "
        f"m* = {season_star}, J_min ≈ {float(J_mat[i_best, j_best]):.6e}"
    )

    return alpha_star, season_star, J_mat


# ============================================================
# 5. Plots
# ============================================================

def plot_J_heatmap(
    alpha_grid: np.ndarray,
    season_grid: Sequence[int],
    J_mat: np.ndarray,
    alpha_star: float,
    season_star: int,
    save_path: Optional[str] = None,
):
    """
    Heatmap of J(alpha, m) with the best pair marked.
    """
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    season_grid = np.asarray(season_grid, dtype=int)

    plt.figure()
    ax = plt.gca()

    # imshow wants matrix [row, col] => [season_idx, alpha_idx]
    im = ax.imshow(
        J_mat,
        aspect="auto",
        origin="lower",
        extent=[
            alpha_grid[0], alpha_grid[-1],
            season_grid[0] - 0.5, season_grid[-1] + 0.5
        ],
    )
    plt.colorbar(im, ax=ax, label=r"$J(s, m)$ (MSE validación)")

    ax.set_xlabel(r"Suavización $s=\alpha$")
    ax.set_ylabel(r"Periodo estacional $m$")
    ax.set_title(r"Mapa de calor de $J(s, m)$ en validación")

    # Mark best pair
    ax.scatter(
        [alpha_star],
        [season_star],
        color="red",
        edgecolor="black",
        zorder=5,
        label=fr"Mejor: $(s^\star, m^\star)=({alpha_star:.3f}, {season_star})$",
    )
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_logprice_vs_forecast(
    t: np.ndarray,
    Z: np.ndarray,
    fitted_train: np.ndarray,
    forecast_future: np.ndarray,
    N_train: int,
    N_val: int,
    meta: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Plot log-price vs forecast:

    - Train: fitted Holt–Winters values (smoothed).
    - Val+Test: pure forecasts from last training level/trend/season.
    """
    Z = np.asarray(Z, dtype=float).ravel()
    fitted_train = np.asarray(fitted_train, dtype=float).ravel()
    forecast_future = np.asarray(forecast_future, dtype=float).ravel()

    N = Z.size
    N_future = N_val + (N - N_train - N_val)

    assert fitted_train.size == N_train
    assert forecast_future.size == N_future
    assert N_train + N_future == N

    forecast_all = np.concatenate([fitted_train, forecast_future])

    plt.figure()
    ax = plt.gca()

    ax.plot(t, Z, label=r"$\log(\text{precio})$", linewidth=1.5)
    ax.plot(t, forecast_all,
            label=r"Pronóstico Holt–Winters (train + extrapolación)",
            linewidth=1.5)

    # Vertical boundaries
    ax.axvline(N_train - 0.5, color="k", linestyle="--", linewidth=1)
    ax.axvline(N_train + N_val - 0.5, color="k", linestyle="--", linewidth=1)

    # Segment labels
    N_test = N - N_train - N_val
    ymax = max(np.max(Z), np.max(forecast_all))
    ymin = min(np.min(Z), np.min(forecast_all))
    ytext = ymin + 0.05 * (ymax - ymin)

    ax.text(N_train / 2.0, ytext, "Train", ha="center", va="bottom")
    ax.text(N_train + N_val / 2.0, ytext, "Validation", ha="center", va="bottom")
    ax.text(N_train + N_val + N_test / 2.0, ytext, "Test", ha="center", va="bottom")

    ax.set_xlabel("Tiempo (índice)")
    ax.set_ylabel("Log-precio")
    ax.set_title(
        f"{meta['ticker']} – Holt–Winters aditivo (grid en $s$ y $m$)\n"
        r"Serie suavizada en train, extrapolación con tendencia+estacionalidad en validación+prueba"
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
    n_alpha_grid: int = 25,
    season_grid: Sequence[int] = (5, 10, 20),
    gamma: Optional[float] = None,
    figs_dir: Optional[str] = None,
):
    """
    Run the full pipeline for one ticker.

    1) Load log-prices.
    2) Split into train / val / test.
    3) Grid-search (alpha, m) via J(alpha, m) on validation, using ONLY
       models fitted on train.
    4) Refit on train with (alpha*, m*) and build:
         - fitted Holt–Winters values on train,
         - multi-step forecasts (level+trend+season) on val+test.
    5) Plot J(alpha, m) and forecast vs original log-price.

    Parameters
    ----------
    season_grid : list of int
        Candidate seasonal periods. Must satisfy N_train >= 2 * m
        for at least one m to be usable.
    gamma : float or None
        Seasonal smoothing parameter (shared for all alphas) if
        not None. If None, gamma = alpha inside the model.
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

    # 4. Sweep J(alpha, m) using ONLY train fit
    alpha_star, season_star, J_mat = sweep_alpha_season_grid(
        Z_train, Z_val, alpha_grid, season_grid, gamma=gamma
    )

    # 5. Plot J(alpha, m) heatmap
    if figs_dir is not None:
        import os
        os.makedirs(figs_dir, exist_ok=True)
        path_J = f"{figs_dir}/{ticker}_J_heatmap_hw.png"
    else:
        path_J = None
    plot_J_heatmap(alpha_grid, season_grid, J_mat, alpha_star, season_star, save_path=path_J)

    # 6. Refit on train with (alpha*, m*), build fitted train and future forecasts
    fitted_train, last_level, last_trend, seasonal, last_pos = (
        holt_winters_additive_train(
            Z_train, alpha_star, season_star, gamma=(gamma if gamma is not None else None)
        )
    )

    # Future (validation + test) forecasts: multi-step HW from training end
    N_future = N_val + N_test
    m = season_star
    forecast_future = np.zeros(N_future, dtype=float)
    for k in range(N_future):
        h = k + 1  # horizon from last training point
        pos_future = (last_pos + h) % m
        forecast_future[k] = last_level + h * last_trend + seasonal[pos_future]

    # 7. Plot log-price vs forecast
    if figs_dir is not None:
        path_ts = f"{figs_dir}/{ticker}_logprice_vs_forecast_hw_grid.png"
    else:
        path_ts = None
    plot_logprice_vs_forecast(
        t=t,
        Z=Z,
        fitted_train=fitted_train,
        forecast_future=forecast_future,
        N_train=N_train,
        N_val=N_val,
        meta=meta,
        save_path=path_ts,
    )


if __name__ == "__main__":
    # Example:
    #   python holt_winters_grid_alpha_season.py
    main()
