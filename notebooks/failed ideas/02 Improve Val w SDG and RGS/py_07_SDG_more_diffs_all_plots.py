# -*- coding: utf-8 -*-
"""
Guerrero (2007) penalized trend on S&P 500 with SGD-style tuning of s,
and visualizations for ALL d in d_list.

Pipeline:
  1) Load S&P 500 daily log-prices (from yfinance or CSV).
  2) Strict split into train / validation / test.
  3) For each differencing order d in d_list:
       - Optimize smoothness s in (0,1) via SGD-style finite-difference gradient
         on validation MSE, printing MSE each iteration (sanity check).
  4) For each d, plot train/val/test with its best s(d).
"""

from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from math import comb as _comb
import pandas as pd

# ---------- Optional GPU: CuPy (not used in SGD, but kept in core solver) ----------
try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False


# ---------- Core numerical routines (Guerrero-style penalized trend) ----------

def difference_matrix(N: int, d: int) -> np.ndarray:
    """
    Construct K: (N-d) x N with rows implementing the d-th forward difference Δ^d t_t.
    For d = 0, K = I_N (no differencing).
    """
    if d == 0:
        return np.eye(N)
    K = np.zeros((N - d, N), dtype=float)
    coeffs = np.array([(-1) ** (d - k) * _comb(d, k) for k in range(d + 1)], dtype=float)
    for r in range(N - d):
        K[r, r: r + d + 1] = coeffs
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

    Normalization: S_max = 1 - d/N, and we enforce

        s_unit = S_raw(λ) / S_max.
    """
    s_unit = float(s_unit)
    if s_unit <= 0.0:
        return 0.0
    if s_unit >= 1.0:
        s_unit = 0.999999

    if d == 0:
        # closed form
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

    # sanity guard: if λ becomes astronomically large, bail out
    LAM_MAX = 1e8
    if not np.isfinite(lam) or lam > LAM_MAX:
        raise ValueError(f"lambda too large or non-finite (λ={lam}) for d={d}, s={s_unit}")

    if use_gpu and _HAS_CUPY:
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

    # CPU path
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


def forecast_trend(t_hat: np.ndarray, d: int, m_hat: float, h: int) -> np.ndarray:
    """
    h-step-ahead forecast of the trend/mean for arbitrary d >= 0.

    d = 0 : constant trend               t_{N+h} = m_hat
    d = 1 : linear trend                 t_{N+h} = t_N + h * m_hat
    d = 2 : quadratic / HP-style trend
    d >= 3 : higher-order polynomial trend obtained via recursion
             from Δ^d t_t = m_hat.
    """
    t_hat = np.asarray(t_hat, dtype=float).ravel()
    if h <= 0:
        return np.array([], dtype=float)

    if d == 0:
        return np.full(h, m_hat, dtype=float)
    if d < 0:
        raise ValueError("d must be >= 0")

    N = t_hat.size
    d_eff = min(d, N)

    # last d_eff values: [t_{N-d_eff}, ..., t_{N-1}]
    last = t_hat[-d_eff:].copy()

    # coefficients in:
    #   t_t = m_hat - sum_{k=0}^{d_eff-1} (-1)^{d_eff-k} C(d_eff, k) t_{t-d_eff+k}
    coeffs = np.array(
        [(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(d_eff)],
        dtype=float,
    )

    out = np.empty(h, dtype=float)
    for i in range(h):
        sum_prev = float(coeffs @ last)
        next_val = m_hat - sum_prev
        out[i] = next_val
        # shift window
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
    Z_val = Z[N_train:N_train + N_val]
    Z_test = Z[N_train + N_val:]

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

    try:
        t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
            Z_train, d=d, s_unit=s_unit, use_gpu=False
        )
    except Exception:
        return np.inf

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
    verbose: bool = True,
):
    """
    SGD-style optimization of s for fixed d.

    - Z_train, Z_val: 1D arrays.
    - s is constrained to [1e-3, 0.999].
    - gradient is estimated by symmetric finite difference in s.
    - if stochastic=True, each step uses a random subset of validation points.
    - verbose=True prints loss (MSE) each iteration (sanity check).
    """
    rng = np.random.default_rng(random_state)
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()
    N_val = Z_val.size

    def loss_on_subset(s_unit: float, idx_val: Optional[np.ndarray] = None) -> float:
        if N_val == 0:
            return np.nan

        try:
            t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
                Z_train, d=d, s_unit=s_unit, use_gpu=False
            )
        except Exception:
            return np.inf

        t_hat_1d = np.asarray(t_hat, dtype=float).ravel()
        h = N_val
        t_fore_all = forecast_trend(t_hat_1d, d=d, m_hat=m_hat, h=h)

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
        if stochastic and N_val > 0:
            batch_size = max(1, int(batch_frac * N_val))
            idx_val = rng.choice(N_val, size=batch_size, replace=False)
        else:
            idx_val = None

        # current approximate loss
        L = loss_on_subset(s, idx_val=idx_val)

        # sanity: break if loss explodes or is non-finite
        if not np.isfinite(L) or L > 1e12:
            if verbose:
                print(f"[d={d}] iter={k:02d} | s={s:.4f} | loss={L} (non-finite / huge) -> stop")
            break

        # finite-difference gradient
        s_plus = min(0.999, s + eps)
        s_minus = max(1e-3, s - eps)
        L_plus = loss_on_subset(s_plus, idx_val=idx_val)
        L_minus = loss_on_subset(s_minus, idx_val=idx_val)
        g = (L_plus - L_minus) / (s_plus - s_minus)

        if verbose:
            print(
                f"[d={d}] iter={k:02d} | s={s:.4f} | approx_MSE={L:.6e} | "
                f"grad≈{g:.3e}"
            )

        history.append(dict(iter=k, s=s, loss=L, grad=g))

        # gradient step + projection
        s_new = s - lr * g
        s_new = float(np.clip(s_new, 1e-3, 0.999))
        s = s_new

    # evaluate final on full validation (strict)
    final_loss = val_loss_for_d_s(Z_train, Z_val, d=d, s_unit=s)
    return s, final_loss, history


# ---------- Visualization for each d (strict train/val/test) ----------

def plot_d_train_val_test_strict(
    Z_all: np.ndarray,
    N_train: int,
    N_val: int,
    d: int,
    s_unit: float,
    meta: Dict[str, Any],
):
    """
    Strict split visualization for a single d:

      - Fit Guerrero trend with (d, s_unit) on TRAIN only.
      - Forecast over VAL and TEST.
      - Plot Z_t and trend/forecasts for all three segments.
    """
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    N_all = Z_all.size

    Z_train = Z_all[:N_train]
    Z_val = Z_all[N_train:N_train + N_val]
    Z_test = Z_all[N_train + N_val:]
    N_val = Z_val.size
    N_test = Z_test.size

    # fit only on train
    t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
        Z_train, d=d, s_unit=s_unit, use_gpu=False
    )
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    # forecast over val+test
    h_total = N_val + N_test
    t_fore_all = forecast_trend(t_hat, d=d, m_hat=m_hat, h=h_total)
    t_fore_val = t_fore_all[:N_val]
    t_fore_test = t_fore_all[N_val:]

    # RMSEs
    rmse_train = float(np.sqrt(np.mean((t_hat - Z_train) ** 2)))
    rmse_val = float(np.sqrt(np.mean((t_fore_val - Z_val) ** 2))) if N_val > 0 else np.nan
    rmse_test = float(np.sqrt(np.mean((t_fore_test - Z_test) ** 2))) if N_test > 0 else np.nan

    # indices
    idx_all = np.arange(N_all)
    idx_train = idx_all[:N_train]
    idx_val = idx_all[N_train:N_train + N_val]
    idx_test = idx_all[N_train + N_val:]

    plt.figure(figsize=(11, 5))
    ax = plt.gca()

    # observed series
    ax.plot(idx_train, Z_train, label="Z_t (train)", linewidth=1.0)
    if N_val > 0:
        ax.plot(idx_val, Z_val, label="Z_t (val)", linewidth=1.0)
    if N_test > 0:
        ax.plot(idx_test, Z_test, label="Z_t (test)", linewidth=1.0)

    # trend (fit on train)
    ax.plot(idx_train, t_hat, label="trend (fit on train)", linewidth=2.0)

    # forecast trend
    if N_val > 0:
        ax.plot(idx_val, t_fore_val, "--", label="trend forecast (val)", linewidth=2.0)
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
        f"{meta['ticker']} – d={d}, s≈{s_unit:.3f}, λ≈{lam:.3g}, m̂≈{m_hat:.4f}\n"
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
    # 1. Load S&P 500
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
        frac_val=0.2,    # ~20% val, rest test
        min_train=80,
        min_val=60,
    )
    print(f"Train length = {N_train}, val = {N_val}, test = {N_test}.")

    # 3. For each d, run SGD-style optimization over s (with printed MSE)
    d_list = [1, 2, 3, 4]  # you can try 4+ if numerically stable
    best_by_d: Dict[int, Dict[str, Any]] = {}

    for d in d_list:
        print(f"\n=== Optimizing s for d={d} (SGD-style) ===")
        s_star, loss_star, hist = optimize_s_sgd(
            Z_train,
            Z_val,
            d=d,
            s0=0.3,
            n_iter=15,
            lr=0.2,
            eps=5e-3,
            stochastic=True,
            batch_frac=0.5,
            random_state=123 + d,
            verbose=True,  # prints MSE each iteration
        )
        best_by_d[d] = dict(s=s_star, loss=loss_star, history=hist)
        print(f"[summary] d={d}: final best s≈{s_star:.4f}, full-val_MSE≈{loss_star:.6e}")

    # 4. Visualizations for ALL d using its own best s(d)
    for d in d_list:
        s_star = best_by_d[d]["s"]
        loss = best_by_d[d]["loss"]

        if not np.isfinite(loss) or not np.isfinite(s_star):
            print(f"\n--- Skipping d={d}: non-finite best loss or s ---")
            continue

        print(f"\n--- Plotting strict train/val/test for d={d}, s≈{s_star:.4f} ---")
        plot_d_train_val_test_strict(
            Z_all=Z_all,
            N_train=N_train,
            N_val=N_val,
            d=d,
            s_unit=s_star,
            meta=meta,
        )


if __name__ == "__main__":
    main()
