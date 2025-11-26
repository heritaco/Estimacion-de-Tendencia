# -*- coding: utf-8 -*-
"""
Guerrero (2007) penalized trend on S&P 500 with gradient-based 1D optimization in λ.

For each differencing order d:

  1) Precompute K, K', K'K, etc. on the TRAIN segment.
  2) Define validation MSE J_d(λ) = MSE on validation block when fitting on train with λ.
  3) Compute ∂J_d / ∂λ via matrix calculus (m treated as frozen at its fixed-point value).
  4) Optimize λ via gradient descent in θ = log(λ) (1D, with simple clipping).
  5) For each d, plot train/val/test with λ* and the implied smoothness index s*.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import comb as _comb

# ============================================================
#  Global caches
# ============================================================

# Cache of eigenvalues of K K' for (N,d)
_NU_CACHE: Dict[Tuple[int, int], np.ndarray] = {}

# Cache of precomputed matrices on TRAIN for (N_train,d)
_PRECOMP_TRAIN_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}


def _get_nu_for_lambda(N: int, d: int, K: np.ndarray) -> np.ndarray:
    """
    Return eigenvalues of K K' cached for (N, d).
    Only computes eigvalsh(KK') the first time for each (N, d).
    """
    key = (N, d)
    if key in _NU_CACHE:
        return _NU_CACHE[key]

    KKt = K @ K.T
    nu = np.linalg.eigvalsh(KKt)
    nu = np.maximum(nu, 0.0)  # numeric clamp
    _NU_CACHE[key] = nu
    return nu


# ============================================================
#  Guerrero core: difference matrix
# ============================================================

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


# ============================================================
#  Precomputation on TRAIN for a given (N_train,d)
# ============================================================

def get_precomp_train(N: int, d: int) -> Dict[str, Any]:
    """
    Precompute and cache matrices that depend only on (N_train, d):
      - K, K', K'K, K'1, eigenvalues ν of K K'.
    """
    key = (N, d)
    if key in _PRECOMP_TRAIN_CACHE:
        return _PRECOMP_TRAIN_CACHE[key]

    K = difference_matrix(N, d)           # (N-d) x N
    KT = K.T                              # N x (N-d)
    KTK = KT @ K                          # N x N
    N_d = N - d
    ones_d = np.ones(N_d, dtype=float)
    K1 = KT @ ones_d                      # N x 1
    nu = _get_nu_for_lambda(N, d, K)      # eigenvalues of K K'

    pre = dict(
        N=N,
        d=d,
        N_d=N_d,
        K=K,
        KT=KT,
        KTK=KTK,
        K1=K1,
        nu=nu,
    )
    _PRECOMP_TRAIN_CACHE[key] = pre
    return pre


# ============================================================
#  Train-solver for fixed λ, and gradient pieces
# ============================================================

def solve_trend_lambda(
    Z_train: np.ndarray,
    lam: float,
    pre: Dict[str, Any],
    max_fp: int = 80,
    tol_fp: float = 1e-10,
):
    """
    Solve Guerrero penalized trend on TRAIN for a given λ:

        A(λ) t = Z_train + λ m K'1,
        A(λ) = I + λ K'K,

    with fixed-point iteration on m = mean(K t).

    Returns:
        t_hat : (N_train,)
        m_hat : scalar
        L     : Cholesky factor of A (A = L L^T)
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    N = pre["N"]
    d = pre["d"]
    K = pre["K"]
    KTK = pre["KTK"]
    K1 = pre["K1"]

    # Construct A(λ) and its Cholesky
    A = np.eye(N, dtype=float) + lam * KTK
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Cholesky failed for λ={lam}") from e

    # Fixed-point for m
    KZ = K @ Z_train
    m = float(np.mean(KZ))
    for _ in range(max_fp):
        rhs = Z_train + lam * m * K1
        y = np.linalg.solve(L, rhs)
        t = np.linalg.solve(L.T, y)
        Kt = K @ t
        m_new = float(np.mean(Kt))
        if abs(m_new - m) < tol_fp:
            m = m_new
            break
        m = m_new

    return t, m, L


def forecast_trend_and_derivative(
    t_hat: np.ndarray,
    t_prime: np.ndarray,
    d: int,
    m_hat: float,
    h: int,
):
    """
    Compute both:
      - t_fore: h-step-ahead forecast of the trend,
      - t_fore_prime: derivative ∂t_fore/∂λ,
    assuming m_hat is treated as constant.

    Input:
      t_hat   : length N array (trend on train)
      t_prime : length N array (∂t_hat/∂λ) on train
    """
    t_hat = np.asarray(t_hat, dtype=float).ravel()
    t_prime = np.asarray(t_prime, dtype=float).ravel()
    if h <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    N = t_hat.size
    t_fore = np.empty(h, dtype=float)
    t_fore_prime = np.empty(h, dtype=float)

    if d <= 0:
        # d=0: constant trend m_hat, independent of t_hat when m is frozen
        t_fore[:] = m_hat
        t_fore_prime[:] = 0.0
        return t_fore, t_fore_prime

    if d == 1:
        # Trend is linear: t_{N+h} = t_N + h * m_hat
        tN = t_hat[-1]
        tN_prime = t_prime[-1]
        for j in range(h):
            step = j + 1
            t_fore[j] = tN + m_hat * step
            t_fore_prime[j] = tN_prime     # m_hat treated as constant
        return t_fore, t_fore_prime

    # d >= 2: use the general recurrence, which is linear in the last d values
    d_eff = min(d, N)
    last = t_hat[-d_eff:].copy()
    last_prime = t_prime[-d_eff:].copy()
    coeffs = np.array(
        [(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(d_eff)],
        dtype=float,
    )

    for j in range(h):
        sum_prev = float(coeffs @ last)
        next_val = m_hat - sum_prev
        t_fore[j] = next_val

        # derivative: m_hat is constant, so derivative is only through last
        next_prime = -float(coeffs @ last_prime)
        t_fore_prime[j] = next_prime

        # shift window
        last[:-1] = last[1:]
        last[-1] = next_val
        last_prime[:-1] = last_prime[1:]
        last_prime[-1] = next_prime

    return t_fore, t_fore_prime


def val_loss_and_grad_lambda(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    d: int,
    lam: float,
    pre: Dict[str, Any],
):
    """
    Compute validation MSE J_d(λ) and its derivative ∂J_d/∂λ.

    Protocol:
      - Solve Guerrero trend on TRAIN with given λ.
      - Compute t'(λ) by solving A t' = m K'1 - K'K t (m treated as frozen at m_hat).
      - Forecast on VAL and compute derivative of forecast via
        forecast_trend_and_derivative.
      - Return J and dJ/dλ.

    If λ is invalid or the system is ill-conditioned, returns (inf, 0).
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()
    N_val = Z_val.size

    if lam <= 0 or not np.isfinite(lam):
        return np.inf, 0.0
    if N_val == 0:
        return np.nan, 0.0

    N = pre["N"]
    d_pre = pre["d"]
    if d_pre != d:
        raise ValueError("pre['d'] and requested d do not match.")

    K = pre["K"]
    KTK = pre["KTK"]
    K1 = pre["K1"]

    # 1) Solve for t_hat, m_hat, and L
    try:
        t_hat, m_hat, L = solve_trend_lambda(Z_train, lam, pre)
    except np.linalg.LinAlgError:
        return np.inf, 0.0

    # 2) Compute t_prime via:
    #    A t' = m_hat K'1 - K'K t_hat
    rhs_prime = m_hat * K1 - (KTK @ t_hat)
    y = np.linalg.solve(L, rhs_prime)
    t_prime = np.linalg.solve(L.T, y)

    # 3) Forecast and forecast derivative on validation horizon
    h = N_val
    t_fore, t_fore_prime = forecast_trend_and_derivative(t_hat, t_prime, d, m_hat, h)

    # 4) Validation MSE and gradient
    resid = Z_val - t_fore
    J = float(np.mean(resid ** 2))
    grad = float(-2.0 / h * (resid @ t_fore_prime))

    return J, grad


# ============================================================
#  Data: S&P 500 daily
# ============================================================

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


# ============================================================
#  Smoothness index s from λ (for reporting)
# ============================================================

def compute_s_unit_from_lambda(lam: float, pre: Dict[str, Any]) -> float:
    """
    Compute Guerrero's smoothness index s_unit from λ for reporting:

        S_raw(λ) = 1 - (1/N) [ d + sum_i 1/(1+λ ν_i) ],
        s_unit   = S_raw / S_max,  S_max = 1 - d/N.

    For d=0: S_raw = λ/(1+λ), S_max=1, s_unit = λ/(1+λ).
    """
    N = pre["N"]
    d = pre["d"]

    if d == 0:
        S_raw = lam / (1.0 + lam)
        return float(S_raw)

    nu = pre["nu"]
    S_raw = 1.0 - (d + np.sum(1.0 / (1.0 + lam * nu))) / N
    S_max = 1.0 - d / N
    s_unit = S_raw / S_max
    return float(s_unit)


# ============================================================
#  1D gradient descent in θ = log(λ)
# ============================================================

def minimize_lambda_gradient(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    d: int,
    pre: Dict[str, Any],
    lam_init: float = 1.0,
    n_iter: int = 20,
    lr: float = 0.1,
    verbose: bool = True,
):
    """
    Minimize validation MSE J_d(λ) w.r.t λ > 0 using gradient descent in θ = log(λ).

    We do:
        θ_{k+1} = θ_k - lr * (dJ/dθ),
    where dJ/dθ = (dJ/dλ) * λ, and λ = exp(θ).

    We clip θ to [log(λ_min), log(λ_max)] to avoid numerical extremes.
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()

    lam_min = 1e-8
    lam_max = 1e8
    theta_min = np.log(lam_min)
    theta_max = np.log(lam_max)

    lam_init = float(np.clip(lam_init, lam_min, lam_max))
    theta = np.log(lam_init)

    best_J = np.inf
    best_lam = lam_init

    for k in range(n_iter):
        lam = float(np.exp(theta))
        J, dJ_dlam = val_loss_and_grad_lambda(Z_train, Z_val, d, lam, pre)
        if not np.isfinite(J) or not np.isfinite(dJ_dlam):
            if verbose:
                print(f"[d={d}] iter={k:02d} λ={lam:.3e} -> non-finite J or grad, stopping")
            break

        if verbose:
            print(
                f"[d={d}] iter={k:02d} λ={lam:.3e} | "
                f"J_val={J:.6e} | dJ/dλ={dJ_dlam:.3e}"
            )

        if J < best_J:
            best_J = J
            best_lam = lam

        # gradient in θ = log(λ)
        g_theta = dJ_dlam * lam
        theta = theta - lr * g_theta
        theta = float(np.clip(theta, theta_min, theta_max))

    # final evaluation at best_lam
    final_J, _ = val_loss_and_grad_lambda(Z_train, Z_val, d, best_lam, pre)
    return best_lam, final_J


# ============================================================
#  Visualization: train/val/test for a given (d, λ)
# ============================================================

def plot_d_train_val_test_lambda(
    Z_all: np.ndarray,
    N_train: int,
    N_val: int,
    d: int,
    lam: float,
    pre: Dict[str, Any],
    meta: Dict[str, Any],
):
    """
    Strict split visualization for a single d and λ:

      - Fit Guerrero trend with (d, λ) on TRAIN only.
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
    t_hat, m_hat, L = solve_trend_lambda(Z_train, lam, pre)
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    # forecast over val+test
    h_total = N_val + N_test
    # reuse derivative routine with t_prime=0 to get forecast only
    t_prime_zero = np.zeros_like(t_hat)
    t_fore_all, _ = forecast_trend_and_derivative(t_hat, t_prime_zero, d, m_hat, h_total)

    t_fore_val = t_fore_all[:N_val]
    t_fore_test = t_fore_all[N_val:]

    # RMSEs
    rmse_train = float(np.sqrt(np.mean((t_hat - Z_train) ** 2)))
    rmse_val = float(np.sqrt(np.mean((t_fore_val - Z_val) ** 2))) if N_val > 0 else np.nan
    rmse_test = float(np.sqrt(np.mean((t_fore_test - Z_test) ** 2))) if N_test > 0 else np.nan

    # effective smoothness index from λ
    s_unit = compute_s_unit_from_lambda(lam, pre)

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
        f"{meta['ticker']} – d={d}, λ≈{lam:.3g}, s≈{s_unit:.3f}, m̂≈{m_hat:.4f}\n"
        f"RMSE(train)={rmse_train:.4f}, RMSE(val)={rmse_val:.4f}, RMSE(test)={rmse_test:.4f}"
    )
    ax.set_xlabel("t (days)")
    ax.set_ylabel("log-price")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ============================================================
#  Main
# ============================================================

def main():
    # 1. Load S&P 500
    t_all, Z_all, meta = load_sp500_series(
        ticker="^GSPC",
        start="2010-01-01",
        end=None,
        use_log=True,
        csv_path=None,  # set this to a CSV path if you prefer offline data
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

    # 3. For each d, precompute matrices and minimize J_d(λ) via gradient
    d_list = [1, 2, 3]  # you can extend to [1,2,3,4] if stable
    best_by_d: Dict[int, Dict[str, Any]] = {}

    for d in d_list:
        print(f"\n=== d={d}: precomputing matrices on TRAIN ===")
        pre = get_precomp_train(N_train, d)

        print(f"=== d={d}: minimizing validation MSE in λ (gradient in log λ) ===")
        lam_star, J_star = minimize_lambda_gradient(
            Z_train,
            Z_val,
            d=d,
            pre=pre,
            lam_init=1.0,  # you can change this initial guess
            n_iter=20,
            lr=0.05,       # tune step size if needed
            verbose=True,
        )
        s_star = compute_s_unit_from_lambda(lam_star, pre)
        best_by_d[d] = dict(lam=lam_star, loss=J_star, s=s_star)
        print(
            f"[summary] d={d}: λ*≈{lam_star:.6g}, s*≈{s_star:.4f}, "
            f"val_MSE≈{J_star:.6e}"
        )

    # 4. Visualizations for ALL d using its own best λ(d)
    for d in d_list:
        lam_star = best_by_d[d]["lam"]
        loss = best_by_d[d]["loss"]
        pre = get_precomp_train(N_train, d)

        if not np.isfinite(loss) or not np.isfinite(lam_star):
            print(f"\n--- Skipping d={d}: non-finite best loss or λ ---")
            continue

        print(f"\n--- Plotting strict train/val/test for d={d}, λ≈{lam_star:.4f} ---")
        plot_d_train_val_test_lambda(
            Z_all=Z_all,
            N_train=N_train,
            N_val=N_val,
            d=d,
            lam=lam_star,
            pre=pre,
            meta=meta,
        )


if __name__ == "__main__":
    main()
