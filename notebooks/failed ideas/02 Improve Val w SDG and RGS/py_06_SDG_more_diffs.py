# -*- coding: utf-8 -*-
"""
Guerrero (2007) penalized trend on S&P 500 with SGD-style tuning of smoothness s.

Pipeline:
  1) Load S&P 500 daily log-prices (from yfinance or CSV).
  2) Split into train / validation / test.
  3) For each differencing order d in D = {1,2,...}:
       - Optimize s in (0,1) by SGD-style finite-difference gradient on val MSE.
  4) Choose (d*, s*) with lowest validation MSE.
  5) Re-fit Guerrero trend on train+val with (d*, s*).
  6) Evaluate on train / val / test and display a simple matplotlib plot.

Requirements:
  - numpy, matplotlib, pandas
Optional:
  - yfinance       (for automatic download of S&P 500)
  - cupy-cudaXX    (for GPU acceleration via CuPy; optional, not used in SGD here)
"""

from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from math import comb as _comb
import pandas as pd

# ---------- Optional GPU: CuPy ----------
try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False


# ---------- Core numerical routines (Guerrero-style penalized trend) ----------
def difference_matrix(N: int, d: int) -> np.ndarray:
    """
    Construct K: (N-d) x N whose rows implement the d-th forward difference Δ^d t_t.

    For d = 0, K = I_N (no differencing).
    """
    if d == 0:
        return np.eye(N)
    K = np.zeros((N - d, N), dtype=float)
    coeffs = np.array([(-1) ** (d - k) * _comb(d, k) for k in range(d + 1)], dtype=float)
    for r in range(N - d):
        K[r, r : r + d + 1] = coeffs
    return K


def lambda_from_s_unit(
    s_unit: float,
    N: int,
    d: int,
    K: np.ndarray,
    tol: float = 1e-11,
    maxit: int = 80,
) -> float:
    """
    Map smoothness index s_unit ∈ [0,1) → λ solving

        S_raw(λ) = 1 - (1/N) tr[(I + λ K'K)^(-1)].

    For d ≥ 1:
        tr[(I + λ K'K)^(-1)] = d + Σ_{i=1}^{N-d} 1 / (1 + λ ν_i),
    where ν_i are eigenvalues of K K'.

    Normalization: S_max = 1 - d/N, and

        s_unit = S_raw(λ) / S_max.
    """
    s_unit = float(s_unit)
    if s_unit <= 0.0:
        return 0.0
    if s_unit >= 1.0:
        s_unit = 0.999999

    if d == 0:
        # closed form for d = 0
        return s_unit / (1.0 - s_unit)

    KKt = K @ K.T
    nu = np.linalg.eigvalsh(KKt)
    nu = np.maximum(nu, 0.0)

    S_max = 1.0 - d / N
    target = s_unit * S_max

    def S_raw(lmb: float) -> float:
        trAinv = d + np.sum(1.0 / (1.0 + lmb * nu))
        return 1.0 - trAinv / N

    lo, hi = 0.0, 1.0
    while S_raw(hi) < target and hi < 1e16:
        hi *= 10.0

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        Smid = S_raw(mid)
        if abs(Smid - target) < tol:
            return mid
        if Smid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_penalized_trend_dense(
    Z: np.ndarray,
    d: int,
    s_unit: float,
    use_gpu: bool = False,
):
    """
    Dense penalized trend fit (Guerrero 2007):

        t_hat = (I + λ K'K)^{-1} (Z + λ m K' 1),

    with fixed-point iteration in m = mean(K t_hat).

    Returns:
        t_hat       : fitted trend (N,)
        m_hat       : drift parameter
        lam         : λ
        sigma2_hat  : variance estimate from penalized RSS
        Ainv        : (I + λ K'K)^{-1}
        s_unit_real : realized smoothness index
        used_gpu    : bool
    """
    Z = np.asarray(Z, dtype=float).ravel()
    N = Z.size
    K = difference_matrix(N, d)
    lam = lambda_from_s_unit(s_unit, N, d, K)

    if use_gpu and _HAS_CUPY:
        # GPU path (CuPy)
        Zg = _cp.asarray(Z)
        Kg = _cp.asarray(K)
        KTg = Kg.T
        Ag = _cp.eye(N, dtype=_cp.float64) + lam * (KTg @ Kg)
        Ainv_g = _cp.linalg.inv(Ag)
        K1g = KTg @ _cp.ones(N - d, dtype=_cp.float64)

        m = float(_cp.mean(Kg @ Zg))
        for _ in range(120):
            t_hat_g = Ainv_g @ (Zg + lam * m * K1g)
            m_new = float(_cp.mean(Kg @ t_hat_g))
            if abs(m_new - m) < 1e-10:
                m = m_new
                break
            m = m_new

        resid_g = Zg - t_hat_g
        pen_g = (Kg @ t_hat_g) - m
        dof = max(1, N - d - 1)
        sigma2_hat = float((resid_g.T @ resid_g + lam * (pen_g.T @ pen_g)) / dof)
        trAinv = float(_cp.trace(Ainv_g))
        S_raw = 1.0 - trAinv / N
        S_max = 1.0 - d / N
        s_unit_real = S_raw / S_max

        t_hat = _cp.asnumpy(t_hat_g)
        Ainv = _cp.asnumpy(Ainv_g)
        return t_hat, m, lam, sigma2_hat, Ainv, s_unit_real, True

    # CPU path (NumPy)
    KT = K.T
    A = np.eye(N) + lam * (KT @ K)
    Ainv = np.linalg.inv(A)
    K1 = KT @ np.ones(N - d, dtype=float)

    m = float(np.mean(K @ Z))
    for _ in range(120):
        t_hat = Ainv @ (Z + lam * m * K1)
        m_new = float(np.mean(K @ t_hat))
        if abs(m_new - m) < 1e-10:
            m = m_new
            break
        m = m_new

    resid = Z - t_hat
    pen = (K @ t_hat) - m
    dof = max(1, N - d - 1)
    sigma2_hat = float((resid.T @ resid + lam * (pen.T @ pen)) / dof)
    S_raw = 1.0 - np.trace(Ainv) / N
    S_max = 1.0 - d / N
    s_unit_real = S_raw / S_max
    return t_hat, m, lam, sigma2_hat, Ainv, s_unit_real, False


from math import comb as _comb
import numpy as np

def forecast_trend(t_hat: np.ndarray, d: int, m_hat: float, h: int) -> np.ndarray:
    """
    h-step-ahead forecast of the trend/mean for arbitrary d >= 0.

    d = 0 : constant trend               t_{N+h} = m_hat
    d = 1 : linear trend                 t_{N+h} = t_N + h * m_hat
    d = 2 : quadratic / HP-style trend   (same as old closed-form)
    d >= 3 : higher-order polynomial trend obtained by recursion
             from Δ^d t_t = m_hat.
    """
    t_hat = np.asarray(t_hat, dtype=float).ravel()
    if h <= 0:
        return np.array([], dtype=float)

    # d = 0: trivial
    if d == 0:
        return np.full(h, m_hat, dtype=float)

    if d < 0:
        raise ValueError("d must be >= 0")

    N = t_hat.size
    if d > N:
        # Degenerate case: if d is larger than available history,
        # just cap it at N (rare in practice for long financial series).
        d_eff = N
    else:
        d_eff = d

    # We keep the last d_eff values: [t_{N-d_eff}, ..., t_{N-1}]
    last = t_hat[-d_eff:].copy()

    # Coefficients for the recursion:
    #   t_t = m_hat - sum_{k=0}^{d-1} (-1)^{d-k} C(d, k) t_{t-d+k}
    # and last[k] = t_{t-d+k} at each step.
    coeffs = np.array(
        [(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(d_eff)],
        dtype=float
    )

    out = np.empty(h, dtype=float)
    for i in range(h):
        # sum over the previous d_eff values
        sum_prev = float(coeffs @ last)
        next_val = m_hat - sum_prev
        out[i] = next_val

        # shift: drop the oldest, append new value
        last[:-1] = last[1:]
        last[-1] = next_val

    return out





# ---------- Data: S&P 500 daily ----------
def load_sp500_series(
    ticker: str = "^GSPC",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    use_log: bool = True,
    csv_path: Optional[str] = None,
):
    """
    Load daily S&P 500 (or any Yahoo-compatible ticker).

    If csv_path is not None:
      - Expect columns 'Date' and 'Adj Close' or 'Close'.

    Else:
      - Use yfinance to download.

    Returns:
      t    : np.ndarray, indices 0..N-1
      Z    : np.ndarray, log-prices (or prices) of length N
      meta : dict with ticker, start, end, use_log
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date")
    else:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required to download S&P 500.\n"
                "Install with: pip install yfinance\n"
                "Or pass csv_path with local data."
            ) from e
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise RuntimeError("Failed to download S&P 500 data from Yahoo.")

    for col in ["Adj Close", "AdjClose", "Close", "close"]:
        if col in df.columns:
            px = df[col].dropna()
            break
    else:
        raise ValueError("No price column found ('Adj Close' or 'Close').")

    Z = px.to_numpy(dtype=float).ravel()
    if use_log:
        Z = np.log(Z)

    t = np.arange(Z.size, dtype=float)
    meta = dict(
        ticker=ticker,
        start=str(px.index[0].date()),
        end=str(px.index[-1].date()),
        use_log=use_log,
    )
    return t, Z, meta


def train_val_test_split(
    Z: np.ndarray,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    min_train: int = 50,
    min_val: int = 50,
):
    """
    3-way split of 1D series Z into train / validation / test.

    Fractions are approximate; min_train / min_val are enforced.
    """
    Z = np.asarray(Z, dtype=float).ravel()
    N = Z.size

    N_train = max(min_train, int(frac_train * N))
    N_val = max(min_val, int(frac_val * N))

    # ensure at least 1 for test
    if N_train + N_val >= N:
        N_val = max(1, N - 1 - N_train)
        if N_val < min_val:
            N_train = max(min_train, N - 1 - N_val)

    N_test = N - N_train - N_val
    if N_test < 1:
        N_test = 1
        N_val = 0
        N_train = N - N_test

    Z_train = Z[:N_train]
    Z_val = Z[N_train : N_train + N_val]
    Z_test = Z[N_train + N_val :]

    return Z_train, Z_val, Z_test, N_train, N_val, N_test


# ---------- Validation loss and SGD-style optimizer for s ----------
def val_loss_for_d_s(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    d: int,
    s_unit: float,
) -> float:
    """
    Compute validation MSE for a given (d, s_unit).

    Uses full validation set.
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()
    N_val = Z_val.size

    if N_val == 0:
        return np.nan

    t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
        Z_train, d=d, s_unit=s_unit, use_gpu=False
    )
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    t_fore = forecast_trend(t_hat, d=d, m_hat=m_hat, h=N_val)
    mse = float(np.mean((t_fore - Z_val) ** 2))
    return mse


def optimize_s_sgd(
    Z_train,
    Z_val,
    d: int,
    s0: float = 0.8,
    n_iter: int = 20,
    lr: float = 0.2,
    eps: float = 5e-3,
    stochastic: bool = False,
    batch_frac: float = 0.5,
    random_state: Optional[int] = None,
):
    """
    SGD-style optimization of s for fixed d.

    - Z_train, Z_val: 1D arrays.
    - s is constrained to [1e-3, 0.999].
    - gradient is estimated by symmetric finite difference in s.
    - if stochastic=True, each step uses a random subset of validation points.
    """
    rng = np.random.default_rng(random_state)
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()
    N_val = Z_val.size

    def loss_on_subset(s_unit: float, idx_val: Optional[np.ndarray] = None) -> float:
        if N_val == 0:
            return np.nan

        # fit on full train
        t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
            Z_train, d=d, s_unit=s_unit, use_gpu=False
        )
        t_hat = np.asarray(t_hat, dtype=float).ravel()

        h = N_val
        t_fore_all = forecast_trend(t_hat, d=d, m_hat=m_hat, h=h)

        if idx_val is None:
            Z_val_sub = Z_val
            t_fore_sub = t_fore_all
        else:
            Z_val_sub = Z_val[idx_val]
            t_fore_sub = t_fore_all[idx_val]

        mse = float(np.mean((t_fore_sub - Z_val_sub) ** 2))
        return mse

    s = float(s0)
    history: List[Dict[str, float]] = []

    for k in range(n_iter):
        # choose validation subset for stochastic gradient
        if stochastic and N_val > 0:
            batch_size = max(1, int(batch_frac * N_val))
            idx_val = rng.choice(N_val, size=batch_size, replace=False)
        else:
            idx_val = None

        # current loss
        L = loss_on_subset(s, idx_val=idx_val)

        # finite-difference gradient
        s_plus = min(0.999, s + eps)
        s_minus = max(1e-3, s - eps)

        L_plus = loss_on_subset(s_plus, idx_val=idx_val)
        L_minus = loss_on_subset(s_minus, idx_val=idx_val)
        g = (L_plus - L_minus) / (s_plus - s_minus)

        # gradient step + projection
        s_new = s - lr * g
        s_new = float(np.clip(s_new, 1e-3, 0.999))

        history.append(dict(iter=k, s=s, loss=L, grad=g))
        s = s_new

    # evaluate final on full validation
    final_loss = val_loss_for_d_s(Z_train, Z_val, d=d, s_unit=s)
    return s, final_loss, history


# ---------- Simple visualization for train / val / test ----------
def plot_best_on_train_val_test_strict(
    Z_all: np.ndarray,
    N_train: int,
    N_val: int,
    d_best: int,
    s_best: float,
    meta: Dict[str, Any],
):
    """
    Strict split:
      - Fit only on TRAIN with (d_best, s_best).
      - Forecast over VAL and TEST.
      - RMSE(val) and RMSE(test) are truly out-of-sample.
    """
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    N_all = Z_all.size

    Z_train = Z_all[:N_train]
    Z_val   = Z_all[N_train:N_train + N_val]
    Z_test  = Z_all[N_train + N_val:]
    N_val   = Z_val.size
    N_test  = Z_test.size

    # --- fit only on TRAIN ---
    t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
        Z_train, d=d_best, s_unit=s_best, use_gpu=False
    )
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    # --- forecasts for VAL+TEST in one shot to keep horizon consistent ---
    h_total = N_val + N_test
    t_fore_all = forecast_trend(t_hat, d=d_best, m_hat=m_hat, h=h_total)
    t_fore_val  = t_fore_all[:N_val]
    t_fore_test = t_fore_all[N_val:]

    # --- RMSEs ---
    rmse_train = float(np.sqrt(np.mean((t_hat - Z_train) ** 2)))
    rmse_val   = float(np.sqrt(np.mean((t_fore_val  - Z_val)  ** 2))) if N_val  > 0 else np.nan
    rmse_test  = float(np.sqrt(np.mean((t_fore_test - Z_test) ** 2))) if N_test > 0 else np.nan

    # --- plotting ---
    idx_all   = np.arange(N_all)
    idx_train = idx_all[:N_train]
    idx_val   = idx_all[N_train:N_train + N_val]
    idx_test  = idx_all[N_train + N_val:]

    plt.figure(figsize=(11, 5))
    ax = plt.gca()

    # observed series
    ax.plot(idx_train, Z_train, label="Z_t (train)", linewidth=1.0)
    if N_val > 0:
        ax.plot(idx_val,   Z_val,   label="Z_t (val)",   linewidth=1.0)
    if N_test > 0:
        ax.plot(idx_test,  Z_test,  label="Z_t (test)",  linewidth=1.0)

    # fitted trend on train
    ax.plot(idx_train, t_hat, label="trend (fit on train)", linewidth=2.0)

    # forecast trend on val and test
    if N_val > 0:
        ax.plot(idx_val,  t_fore_val,  "--", label="trend forecast (val)",  linewidth=2.0)
    if N_test > 0:
        ax.plot(idx_test, t_fore_test, "--", label="trend forecast (test)", linewidth=2.0)

    # boundaries
    ax.axvline(N_train - 0.5, color="k", linestyle="--", linewidth=1)
    ax.text(N_train - 0.5, ax.get_ylim()[1], "train | val",
            ha="right", va="top", fontsize=8)
    if N_val > 0:
        ax.axvline(N_train + N_val - 0.5, color="k", linestyle=":", linewidth=1)
        ax.text(N_train + N_val - 0.5, ax.get_ylim()[1], "val | test",
                ha="right", va="top", fontsize=8)

    ax.set_title(
        f"Guerrero trend on {meta['ticker']} (strict train/val/test)\n"
        f"d={d_best}, s≈{s_best:.3f}, λ≈{lam:.3g}, m̂≈{m_hat:.4f}, "
        f"RMSE(train)={rmse_train:.4f}, RMSE(val)={rmse_val:.4f}, RMSE(test)={rmse_test:.4f}"
    )
    ax.set_xlabel("t (days)")
    ax.set_ylabel("log-price")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()



# ---------- Main ----------
def main():
    # 1. Load S&P 500 (you can change ticker/start/end/csv_path)
    t_all, Z_all, meta = load_sp500_series(
        ticker="^GSPC",
        start="2010-01-01",
        end=None,
        use_log=True,
        csv_path=None,  # set to a CSV path if you prefer offline data
    )

    print(
        f"Loaded {meta['ticker']} from {meta['start']} to {meta['end']} "
        f"(N={Z_all.size}, log={meta['use_log']})."
    )

    # 2. Train / validation / test split
    Z_train, Z_val, Z_test, N_train, N_val, N_test = train_val_test_split(
        Z_all,
        frac_train=0.6,  # ~60% train
        frac_val=0.2,    # ~20% validation, ~20% test
        min_train=80,
        min_val=60,
    )
    print(f"Train length = {N_train}, val = {N_val}, test = {N_test}.")

    # 3. For each d, run SGD-style optimization over s
    d_list = list(range(1, 7))  # d = 1,2,3,4,5,6
    best_by_d: Dict[int, Dict[str, Any]] = {}

    for d in d_list:
        print(f"\n=== Optimizing s for d={d} (SGD-style) ===")
        s_star, loss_star, hist = optimize_s_sgd(
            Z_train,
            Z_val,
            d=d,
            s0=0.8,          # initial s
            n_iter=15,       # number of SGD steps
            lr=0.2,          # learning rate
            eps=5e-3,        # finite-difference step in s
            stochastic=True, # True -> use random subset of validation each step
            batch_frac=0.5,
            random_state=123 + d,
        )
        best_by_d[d] = dict(s=s_star, loss=loss_star, history=hist)
        print(f"d={d}: best s≈{s_star:.4f}, val_MSE≈{loss_star:.6f}")

    # 4. Choose best d by final validation MSE
    best_d = min(best_by_d.keys(), key=lambda dd: best_by_d[dd]["loss"])
    best_s = best_by_d[best_d]["s"]
    print(
        f"\nBest hyperparameters by validation MSE: "
        f"d*={best_d}, s*≈{best_s:.4f}, val_MSE≈{best_by_d[best_d]['loss']:.6f}"
    )

    # 5. Simple plot showing train/val/test behavior of (d*, s*)
    plot_best_on_train_val_test_strict(
        Z_all,
        N_train=N_train,
        N_val=N_val,
        d_best=best_d,
        s_best=best_s,
        meta=meta,
    )


if __name__ == "__main__":
    main()
