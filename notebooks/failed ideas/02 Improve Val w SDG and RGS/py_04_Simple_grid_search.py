# -*- coding: utf-8 -*-
"""
Guerrero (2007) penalized trend on S&P 500 daily data.

- 3-way split: train / validation / test
- Grid of (d, s) hyperparameters:
    * fit on train
    * pick best via RMSE on validation
- Evaluate the best configuration on test
- Simple matplotlib plot for train + val + test

Requirements:
  - Python 3.9+
  - numpy, matplotlib, pandas

Optional:
  - yfinance    (for automatic download of S&P 500)
  - cupy-cudaXX (for GPU acceleration via CuPy; optional)
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
    """K: (N-d) x N; rows implement Δ^d t_t for t = d..N-1."""
    if d == 0:
        return np.eye(N)
    K = np.zeros((N - d, N), dtype=float)
    coeffs = np.array([(-1) ** (d - k) * _comb(d, k) for k in range(d + 1)], dtype=float)
    for r in range(N - d):
        K[r, r:r + d + 1] = coeffs
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
    Map smoothness index s_unit ∈ [0,1) → λ by solving

        S_raw(λ) = 1 - (1/N) tr[(I + λ K'K)^(-1)].

    For d ≥ 1:

        tr[(I + λ K'K)^(-1)] = d + Σ_{i=1}^{N-d} 1 / (1 + λ ν_i),

    where ν_i are eigenvalues of K K'.

    We normalize by S_max = 1 - d/N, and specify

        s_unit = S_raw(λ) / S_max.
    """
    s_unit = float(s_unit)
    if s_unit <= 0.0:
        return 0.0
    if s_unit >= 1.0:
        s_unit = 0.999999

    if d == 0:
        # closed form for d=0
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
    Dense penalized trend fit:

        t_hat = (I + λ K'K)^{-1} (Z + λ m K' 1),

    with fixed point iteration in m = mean(K t_hat).

    Returns:
        t_hat       : fitted trend (N,)
        m_hat       : "drift" parameter
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


def forecast_trend(t_hat: np.ndarray, d: int, m_hat: float, h: int) -> np.ndarray:
    """
    h-step-ahead forecast of the trend/mean.

    d=0: constant
    d=1: linear trend
    d=2: quadratic/HP-style trend
    """
    t_hat = np.asarray(t_hat, dtype=float).ravel()
    if h <= 0:
        return np.array([], dtype=float)
    if d == 0:
        return np.full(h, m_hat, dtype=float)
    if d == 1:
        return t_hat[-1] + m_hat * np.arange(1, h + 1, dtype=float)
    if d == 2:
        tN, tNm1 = t_hat[-1], t_hat[-2]
        hvec = np.arange(1, h + 1, dtype=float)
        return 0.5 * hvec * (hvec + 1) * m_hat + (hvec + 1) * tN - hvec * tNm1
    return np.full(h, t_hat[-1], dtype=float)


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

    # Ensure we leave at least 1 point for test
    if N_train + N_val >= N:
        # compress val if needed
        N_val = max(1, N - 1 - N_train)
        if N_val < min_val:
            # compress train if still too big
            N_train = max(min_train, N - 1 - N_val)

    N_test = N - N_train - N_val
    if N_test < 1:
        # extreme case; just give up on val
        N_test = 1
        N_val = 0
        N_train = N - N_test

    Z_train = Z[:N_train]
    Z_val = Z[N_train:N_train + N_val]
    Z_test = Z[N_train + N_val:]

    return Z_train, Z_val, Z_test, N_train, N_val, N_test


# ---------- Batch evaluation of parameter configurations ----------
def evaluate_config(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    d: int,
    s_unit: float,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Fit penalized trend with (d, s_unit) on training data,
    forecast into validation window, and compute RMSEs.
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()
    N_val = Z_val.size

    t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
        Z_train,
        d=d,
        s_unit=s_unit,
        use_gpu=use_gpu,
    )
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    if N_val > 0:
        t_fore_val = forecast_trend(t_hat, d=d, m_hat=m_hat, h=N_val)
        rmse_val = float(np.sqrt(np.mean((t_fore_val - Z_val) ** 2)))
    else:
        rmse_val = float("nan")

    rmse_train = float(np.sqrt(np.mean((t_hat - Z_train) ** 2)))

    return dict(
        d=d,
        s_unit=s_unit,
        s_real=s_real,
        lam=lam,
        m_hat=m_hat,
        sigma2_hat=sigma2_hat,
        rmse_train=rmse_train,
        rmse_val=rmse_val,
        used_gpu=used_gpu,
    )


def run_preselected_experiments(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    configs: List[Dict[str, Any]],
    use_gpu_default: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run a list of configurations on (Z_train, Z_val).
    """
    results: List[Dict[str, Any]] = []
    for cfg in configs:
        name = cfg.get("name", "unnamed")
        d = int(cfg["d"])
        s = float(cfg["s"])
        use_gpu = bool(cfg.get("use_gpu", use_gpu_default))

        print(f"\n=== Evaluating config {name} (d={d}, s={s:.3f}, GPU={use_gpu}) ===")
        res = evaluate_config(Z_train, Z_val, d=d, s_unit=s, use_gpu=use_gpu)
        res["name"] = name
        results.append(res)

        print(
            f"  -> s_real≈{res['s_real']:.4f}, λ≈{res['lam']:.3g}, "
            f"RMSE_train={res['rmse_train']:.5f}, RMSE_val={res['rmse_val']:.5f}"
        )
    return results


def print_results_table(results: List[Dict[str, Any]]):
    """Print compact table of train/val RMSEs."""
    if not results:
        print("No results to display.")
        return

    header = (
        f"{'name':15s} {'d':>2s} {'s':>6s} {'s_real':>8s} "
        f"{'lambda':>9s} {'RMSE_train':>12s} {'RMSE_val':>10s}"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['name']:15s} "
            f"{r['d']:2d} "
            f"{r['s_unit']:6.3f} "
            f"{r['s_real']:8.4f} "
            f"{r['lam']:9.3g} "
            f"{r['rmse_train']:12.5f} "
            f"{r['rmse_val']:10.5f}"
        )


def find_best_by_val(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return configuration with smallest RMSE_val."""
    finite_results = [r for r in results if np.isfinite(r["rmse_val"])]
    if not finite_results:
        raise RuntimeError("No finite validation RMSE found.")
    return min(finite_results, key=lambda r: r["rmse_val"])


# ---------- Simple visualization for train / val / test ----------
def plot_best_on_train_val_test(
    Z_all: np.ndarray,
    N_train: int,
    N_val: int,
    best_cfg: Dict[str, Any],
    title_prefix: str = "Guerrero trend on S&P 500",
):
    """
    Simple matplotlib figure:

    - Z_t in train / val / test
    - Trend fitted on train
    - Forecast on val and test
    """
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    N_all = Z_all.size

    Z_train = Z_all[:N_train]
    Z_val = Z_all[N_train:N_train + N_val]
    Z_test = Z_all[N_train + N_val:]
    N_val = Z_val.size
    N_test = Z_test.size

    d = best_cfg["d"]
    s = best_cfg["s_unit"]

    # Fit only on train (as in validation step)
    t_hat, m_hat, lam, sigma2_hat, Ainv, s_real, used_gpu = fit_penalized_trend_dense(
        Z_train, d=d, s_unit=s, use_gpu=best_cfg.get("used_gpu", False)
    )
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    # Forecast over validation + test
    h_total = N_val + N_test
    t_fore_all = forecast_trend(t_hat, d=d, m_hat=m_hat, h=h_total)
    t_fore_val = t_fore_all[:N_val]
    t_fore_test = t_fore_all[N_val:]

    # RMSEs
    rmse_train = float(np.sqrt(np.mean((t_hat - Z_train) ** 2)))
    rmse_val = float(np.sqrt(np.mean((t_fore_val - Z_val) ** 2))) if N_val > 0 else np.nan
    rmse_test = float(np.sqrt(np.mean((t_fore_test - Z_test) ** 2))) if N_test > 0 else np.nan

    # Indices
    idx_all = np.arange(N_all)
    idx_train = idx_all[:N_train]
    idx_val = idx_all[N_train:N_train + N_val]
    idx_test = idx_all[N_train + N_val:]

    # Simple figure
    plt.figure(figsize=(11, 5))
    ax = plt.gca()

    # Observations
    ax.plot(idx_train, Z_train, label="Z_t (train)", linewidth=1.0)
    if N_val > 0:
        ax.plot(idx_val, Z_val, label="Z_t (val)", linewidth=1.0)
    if N_test > 0:
        ax.plot(idx_test, Z_test, label="Z_t (test)", linewidth=1.0)

    # Trend (train)
    ax.plot(idx_train, t_hat, label=f"t̂_t (train, s≈{s_real:.2f})", linewidth=2.0)

    # Forecast on val and test
    if N_val > 0:
        ax.plot(idx_val, t_fore_val, "--", label="trend forecast (val)", linewidth=2.0)
    if N_test > 0:
        ax.plot(idx_test, t_fore_test, "--", label="trend forecast (test)", linewidth=2.0)

    # Boundaries
    ax.axvline(N_train - 0.5, color="k", linestyle="--", linewidth=1)
    ax.text(N_train - 0.5, ax.get_ylim()[1], "train | val+test",
            ha="right", va="top", fontsize=8)
    if N_val > 0:
        ax.axvline(N_train + N_val - 0.5, color="k", linestyle=":", linewidth=1)
        ax.text(N_train + N_val - 0.5, ax.get_ylim()[1], "val | test",
                ha="right", va="top", fontsize=8)

    ax.set_title(
        f"{title_prefix}\n"
        f"d={d}, s={s:.3f}, λ≈{lam:.3g}, m̂≈{m_hat:.4f}, "
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
        csv_path=None,
    )

    print(
        f"Loaded {meta['ticker']} from {meta['start']} to {meta['end']} "
        f"(N={Z_all.size}, log={meta['use_log']})."
    )

    # 2. Train / validation / test split
    Z_train, Z_val, Z_test, N_train, N_val, N_test = train_val_test_split(
        Z_all,
        frac_train=0.6,  # you can adjust
        frac_val=0.2,    # remaining is test
        min_train=50,
        min_val=50,
    )
    print(f"Train length = {N_train}, val = {N_val}, test = {N_test}.")

    # 3. Grid of (d, s)
    configs = [
        {"name": "d1_s0.30", "d": 1, "s": 0.30},
        {"name": "d1_s0.90", "d": 1, "s": 0.90},
        {"name": "d1_s0.98", "d": 1, "s": 0.98},
        {"name": "d2_s0.30", "d": 2, "s": 0.30},
        {"name": "d2_s0.90", "d": 2, "s": 0.90},
        {"name": "d2_s0.98", "d": 2, "s": 0.98},
    ]

    # 4. Run experiments on train/val
    results = run_preselected_experiments(Z_train, Z_val, configs, use_gpu_default=False)

    # 5. Summary table and best-by-validation
    print_results_table(results)
    best = find_best_by_val(results)
    print(
        "\nBest by RMSE_val:\n"
        f"  name={best['name']}, d={best['d']}, s={best['s_unit']:.3f}, "
        f"RMSE_val={best['rmse_val']:.5f}, λ≈{best['lam']:.3g}"
    )

    # 6. Simple plot with train / val / test using the best config
    plot_best_on_train_val_test(
        Z_all,
        N_train=N_train,
        N_val=N_val,
        best_cfg=best,
        title_prefix=f"Guerrero trend on {meta['ticker']} (train/val/test)",
    )


if __name__ == "__main__":
    main()
