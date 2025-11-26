# -*- coding: utf-8 -*-
"""
Guerrero (2007) penalized trend on S&P 500 with spectral solver.

For each differencing order d:

  - Define validation loss J_d(s) for smoothness index s∈(0,1).
  - Scan J_d(s) on a grid; detect ALL local minima in s.
  - Refine each local minimum with 1D golden-section.
  - For each local minimum, plot train/val/test fit.

"""

from typing import Callable, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import comb as _comb


# ============================================================
#  Difference matrix
# ============================================================

def difference_matrix(N: int, d: int) -> np.ndarray:
    """
    Construct K: (N-d) x N implementing the d-th forward difference:

        (K τ)_t = Δ^d τ_t,  t = d,...,N-1.

    For d = 0, return I_N (no differencing).
    """
    if d == 0:
        return np.eye(N)
    K = np.zeros((N - d, N), dtype=float)
    coeffs = np.array([(-1) ** (d - k) * _comb(d, k) for k in range(d + 1)], dtype=float)
    for r in range(N - d):
        K[r, r: r + d + 1] = coeffs
    return K


# ============================================================
#  Spectral Guerrero solver: B = K'K = Q Λ Q'
# ============================================================

class GuerreroSpectralSolver:
    """
    Spectral implementation of Guerrero (2007) penalized trend
    for fixed (N, d).

    Precomputes:
      - K: (N-d) x N difference matrix
      - B = K'K: N x N PSD matrix
      - B = Q Λ Q': eigen-decomposition
      - K1 = K' 1 (N-d vector)

    Then for each λ:
      - solves (I + λ B) t = b(λ,m) via Q, Λ in O(N^2).
    """

    def __init__(self, N: int, d: int):
        self.N = int(N)
        self.d = int(d)
        self.K = difference_matrix(N, d)
        self.KT = self.K.T
        # B = K'K
        self.B = self.KT @ self.K                         # N x N, PSD
        # B = Q Λ Q'
        eigvals, Q = np.linalg.eigh(self.B)
        self.eigvals = eigvals                            # (N,)
        self.Q = Q                                        # N x N
        # K1 = K' * 1
        self.K1 = self.KT @ np.ones(N - d, dtype=float)   # (N,)
        # S_max = 1 - d/N (Guerrero normalization)
        self.S_max = 1.0 - d / N

    # ---------- smoothness map: s → λ via trace(A^-1) ----------

    def lambda_from_s(
        self,
        s_unit: float,
        tol: float = 1e-11,
        maxit: int = 80,
    ) -> float:
        """
        Map smoothness index s_unit ∈ [0,1) → λ solving:

            S_raw(λ) = 1 - (1/N) tr[(I + λ K'K)^(-1)],

        with
            tr[(I + λ K'K)^(-1)] = sum_{i=1}^N 1 / (1 + λ λ_i),

        where λ_i are eigenvalues of B = K'K.

        Then s_unit = S_raw(λ) / S_max, with S_max = 1 - d/N.
        """
        s_unit = float(s_unit)
        if s_unit <= 0.0:
            return 0.0
        if s_unit >= 1.0:
            s_unit = 0.999999

        if self.d == 0:
            # closed form when there's no difference penalty
            return s_unit / (1.0 - s_unit)

        target = s_unit * self.S_max
        eigvals = self.eigvals
        N = self.N

        def S_raw(lmb: float) -> float:
            denom = 1.0 + lmb * eigvals
            trAinv = np.sum(1.0 / denom)
            return 1.0 - trAinv / N

        lo, hi = 0.0, 1.0
        # enlarge hi until S_raw(hi) ≥ target
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

    # ---------- solve for a given λ using Q, Λ ----------

    def fit_for_lambda(
        self,
        Z: np.ndarray,
        lam: float,
        m_tol: float = 1e-10,
        max_m_iter: int = 120,
    ):
        """
        Given λ, compute Guerrero trend fit on a series Z of length N:

            t_hat = (I + λ K'K)^(-1) (Z + λ m K'1),

        with fixed-point iteration in m = mean(K t_hat).

        Uses spectral decomposition B = Q Λ Q', so complexity per λ is O(N^2).

        Returns:
            t_hat       : trend fit (N,)
            m_hat       : scalar drift
            sigma2_hat  : variance estimate
            diag_Ainv   : diagonal of (I + λ K'K)^(-1)
            s_unit_real : realized smoothness index S_raw / S_max
        """
        Z = np.asarray(Z, dtype=float).ravel()
        if Z.size != self.N:
            raise ValueError(f"Z length {Z.size} != N {self.N} in solver.")

        N = self.N
        d = self.d
        K = self.K
        Q = self.Q
        eigvals = self.eigvals

        # initial m from raw differences
        m = float(np.mean(K @ Z))

        for _ in range(max_m_iter):
            b = Z + lam * m * self.K1                   # RHS
            y = Q.T @ b                                 # transform to eigenbasis
            denom = 1.0 + lam * eigvals
            z = y / denom                               # elementwise
            t_hat = Q @ z                               # back to original basis

            m_new = float(np.mean(K @ t_hat))
            if abs(m_new - m) < m_tol:
                m = m_new
                break
            m = m_new

        # eigenvalues of A^-1 are α_i = 1 / (1 + λ λ_i)
        alpha = 1.0 / (1.0 + lam * eigvals)
        # diag(A^-1)_j = Σ_i α_i Q_{ji}^2
        diag_Ainv = (Q ** 2) @ alpha

        resid = Z - t_hat
        pen = (K @ t_hat) - m
        dof = max(1, N - d - 1)
        sigma2_hat = float((resid.T @ resid + lam * (pen.T @ pen)) / dof)

        trAinv = np.sum(alpha)
        S_raw = 1.0 - trAinv / N
        s_unit_real = S_raw / self.S_max if self.S_max > 0 else 0.0

        return t_hat, m, sigma2_hat, diag_Ainv, s_unit_real

    # ---------- convenience: fit from s ----------

    def fit_for_s(
        self,
        Z: np.ndarray,
        s_unit: float,
        m_tol: float = 1e-10,
        max_m_iter: int = 120,
    ):
        """
        Given s ∈ (0,1), map to λ and fit Guerrero trend.

        Returns:
            t_hat       : trend fit (N,)
            m_hat       : scalar drift
            lam         : λ
            sigma2_hat  : variance estimate
            diag_Ainv   : diag(A^-1)
            s_unit_real : realized smoothness index
        """
        lam = self.lambda_from_s(s_unit)
        t_hat, m_hat, sigma2_hat, diag_Ainv, s_unit_real = self.fit_for_lambda(
            Z, lam, m_tol=m_tol, max_m_iter=max_m_iter
        )
        return t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_unit_real


# ============================================================
#  Forecast of trend (any d)
# ============================================================

def forecast_trend(t_hat: np.ndarray, d: int, m_hat: float, h: int) -> np.ndarray:
    """
    h-step-ahead forecast of the trend/mean for arbitrary d ≥ 0.

    d = 0 : constant trend
    d = 1 : linear trend
    d ≥ 2 : higher-order polynomial trend implied by Δ^d t_t = m_hat.
    """
    t_hat = np.asarray(t_hat, dtype=float).ravel()
    if h <= 0:
        return np.array([], dtype=float)

    if d == 0:
        return np.full(h, m_hat, dtype=float)
    if d < 0:
        raise ValueError("d must be ≥ 0")

    N = t_hat.size
    d_eff = min(d, N)

    last = t_hat[-d_eff:].copy()  # last d_eff values of trend

    # coefficients in:
    #   t_t = m_hat - Σ_{k=0}^{d_eff-1} (-1)^{d_eff-k} C(d_eff,k) t_{t-d_eff+k}
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


# ============================================================
#  Data: S&P 500 daily
# ============================================================

def load_sp500_series(
    ticker: str = "^GSPC",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    use_log: bool = True,
    csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load daily S&P 500 (or any Yahoo-compatible ticker).

    If csv_path is not None:
      - Expect columns 'Date' and 'Adj Close' or 'Close'.

    Else:
      - Use yfinance to download.

    Returns:
      t    : indices 0..N-1
      Z    : log-prices (or prices) of length N
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
                "yfinance is required; install with `pip install yfinance`, "
                "or pass csv_path with local data."
            ) from e
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise RuntimeError("Failed to download data from Yahoo Finance.")

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
#  Local golden-section and scan for all local minima in s
# ============================================================

def golden_local(
    J: Callable[[float], float],
    a: float,
    b: float,
    n_iter: int = 20,
) -> Tuple[float, float]:
    """
    Golden-section search restricted to [a,b], assuming J is
    unimodal on that interval. Returns (s_star, J(s_star)).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    c = b - invphi * (b - a)
    d_s = a + invphi * (b - a)
    f_c = J(c)
    f_d = J(d_s)

    for _ in range(n_iter):
        if f_c < f_d:
            b = d_s
            d_s = c
            f_d = f_c
            c = b - invphi * (b - a)
            f_c = J(c)
        else:
            a = c
            c = d_s
            f_c = f_d
            d_s = a + invphi * (b - a)
            f_d = J(d_s)

    s_star = 0.5 * (a + b)
    J_star = J(s_star)
    return s_star, J_star


def find_all_local_minima_s(
    J: Callable[[float], float],
    d: int,
    s_min: float = 1e-3,
    s_max: float = 0.999,
    n_grid: int = 300,
    refine: bool = True,
    refine_iter: int = 20,
    verbose: bool = True,
    plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate all local minima of J(s) on [s_min, s_max].

    Steps:
      1) Evaluate J on a grid of n_grid points.
      2) Detect indices i where J[i] <= J[i-1], J[i] <= J[i+1].
      3) (Optional) Refine each candidate via golden-section on
         a small bracket around it.
    """
    # 1) coarse grid
    s_grid = np.linspace(s_min, s_max, n_grid)
    J_grid = np.array([J(s) for s in s_grid])
    # clean version for comparisons (NaN/inf → +inf)
    J_cmp = np.array(J_grid)
    J_cmp[~np.isfinite(J_cmp)] = np.inf

    # 2) discrete local minima
    idx_candidates = []

    for i in range(1, n_grid - 1):
        if not np.isfinite(J_cmp[i]):
            continue
        if (J_cmp[i] <= J_cmp[i - 1]) and (J_cmp[i] <= J_cmp[i + 1]):
            idx_candidates.append(i)

    # endpoints as possible minima
    if np.isfinite(J_cmp[0]) and J_cmp[0] <= J_cmp[1]:
        idx_candidates.append(0)
    if np.isfinite(J_cmp[-1]) and J_cmp[-1] <= J_cmp[-2]:
        idx_candidates.append(n_grid - 1)

    # 3) refine each candidate
    s_list = []
    J_list = []

    for idx in idx_candidates:
        s0 = s_grid[idx]

        if refine:
            if 0 < idx < n_grid - 1:
                a = s_grid[idx - 1]
                b = s_grid[idx + 1]
            elif idx == 0:
                a = s_grid[0]
                b = s_grid[1]
            else:  # idx == n_grid - 1
                a = s_grid[-2]
                b = s_grid[-1]

            s_star, J_star = golden_local(J, a, b, n_iter=refine_iter)
        else:
            s_star = s0
            J_star = J_grid[idx]

        s_list.append(s_star)
        J_list.append(J_star)

    # convert to arrays and sort by s
    s_minima = np.array(s_list, dtype=float)
    J_minima = np.array(J_list, dtype=float)
    order = np.argsort(s_minima)
    s_minima = s_minima[order]
    J_minima = J_minima[order]

    if verbose:
        print(f"\n[d={d}] found {len(s_minima)} local minima on "
              f"[{s_min:.3g},{s_max:.3g}]:")
        for s_star, J_star in zip(s_minima, J_minima):
            print(f"  s≈{s_star:.6f}, J(s)≈{J_star:.6e}")

    if plot:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.plot(s_grid, J_grid, label="J(s) (val MSE)", linewidth=1.5)
        ax.scatter(s_minima, J_minima, s=60, marker="*", color="k",
                   label="local minima")
        ax.set_title(f"All local minima of J(s) for d={d}")
        ax.set_xlabel("s")
        ax.set_ylabel("J(s) (validation MSE)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return s_minima, J_minima


# ============================================================
#  Visualization per d with strict train/val/test protocol
# ============================================================

def plot_d_train_val_test_strict(
    Z_all: np.ndarray,
    N_train: int,
    N_val: int,
    d: int,
    s_unit: float,
    solver: GuerreroSpectralSolver,
    meta: Dict[str, Any],
):
    """
    Strict split visualization for one d:

      - Fit Guerrero trend with (d, s_unit) on TRAIN via solver (t_hat).
      - Build the global Δ^d-polynomial implied by m_hat, anchored on
        the right edge of train.
      - Forecast on VAL+TEST is just the polynomial on those segments.
      - Plot:
          * Z_t (train/val/test),
          * t_hat on train (solid),
          * full polynomial on all t (dashed).
    """
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    N_all = Z_all.size

    Z_train = Z_all[:N_train]
    Z_val = Z_all[N_train:N_train + N_val]
    Z_test = Z_all[N_train + N_val:]
    N_val = Z_val.size
    N_test = Z_test.size

    # --- Guerrero fit on TRAIN ---
    t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s_unit)
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    # --- global Δ^d polynomial (dashed) over whole sample ---
    poly_full = build_polynomial_from_train_tail(
        t_hat_train=t_hat,
        d=d,
        m_hat=m_hat,
        N_total=N_all,
        N_train=N_train,
    )

    # --- RMSEs (train, val, test) ---
    idx_all = np.arange(N_all)
    idx_train = idx_all[:N_train]
    idx_val = idx_all[N_train:N_train + N_val]
    idx_test = idx_all[N_train + N_val:]

    rmse_train = float(np.sqrt(np.mean((t_hat - Z_train) ** 2)))

    rmse_val = (
        float(np.sqrt(np.mean((poly_full[idx_val] - Z_val) ** 2)))
        if N_val > 0 else np.nan
    )
    rmse_test = (
        float(np.sqrt(np.mean((poly_full[idx_test] - Z_test) ** 2)))
        if N_test > 0 else np.nan
    )

    # --- Plot ---
    plt.figure(figsize=(11, 5))
    ax = plt.gca()

    # observed series
    ax.plot(idx_train, Z_train, label="Z_t (train)", linewidth=1.0)
    if N_val > 0:
        ax.plot(idx_val, Z_val, label="Z_t (val)", linewidth=1.0)
    if N_test > 0:
        ax.plot(idx_test, Z_test, label="Z_t (test)", linewidth=1.0)

    # Guerrero trend on train
    ax.plot(idx_train, t_hat, label="trend (fit on train)", linewidth=2.0)

    # full Δ^d-polynomial (same one that generates the forecast), dashed
    ax.plot(
        idx_all,
        poly_full,
        linestyle="--",
        linewidth=2.0,
        label=r"Δ$^d$ polynomial (from train tail)",
    )

    # vertical split markers
    ax.axvline(N_train - 0.5, color="k", linestyle="--", linewidth=1)
    ax.text(N_train - 0.5, ax.get_ylim()[1], "train | val",
            ha="right", va="top", fontsize=8)
    if N_val > 0:
        ax.axvline(N_train + N_val - 0.5, color="k", linestyle=":", linewidth=1)
        ax.text(N_train + N_val - 0.5, ax.get_ylim()[1], "val | test",
                ha="right", va="top", fontsize=8)

    ax.set_title(
        f"{meta['ticker']} – d={d}, s≈{s_unit:.3f}, λ≈{lam:.3g}, m̂≈{m_hat:.4f}\n"
        f"RMSE(train)={rmse_train:.4f}, RMSE(val)={rmse_val:.4f}, "
        f"RMSE(test)={rmse_test:.4f}"
    )
    ax.set_xlabel("t (days)")
    ax.set_ylabel("log-price")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()








def build_polynomial_from_train_tail(
    t_hat_train: np.ndarray,
    d: int,
    m_hat: float,
    N_total: int,
    N_train: int,
) -> np.ndarray:
    """
    Build the global Δ^d polynomial implied by Guerrero's drift m_hat,
    anchored at the *right* edge of the training trend t_hat_train.

    It enforces Δ^d t_t = m_hat for all t, and matches the last d
    points of t_hat_train. For d>=1 this is a degree-d polynomial in t.

    Returns:
        poly_full : array of length N_total with the polynomial values
                    at indices 0..N_total-1.
    """
    t_hat_train = np.asarray(t_hat_train, dtype=float).ravel()
    N_train = int(N_train)
    N_total = int(N_total)

    if d <= 0:
        # trivial "polynomial": constant at m_hat
        return np.full(N_total, float(m_hat), dtype=float)

    d_eff = min(d, N_train)
    poly = np.empty(N_total, dtype=float)

    # right boundary: last index of train
    j0 = N_train - 1

    # anchor last d_eff training points
    start_tail = j0 - d_eff + 1
    poly[start_tail:j0 + 1] = t_hat_train[-d_eff:]

    # ----- backward recursion (t_j from t_{j+1..j+d_eff}) -----
    # coefficients for k=1..d_eff in:
    #   sum_{k=0}^d_eff (-1)^{d_eff-k} C(d_eff,k) t_{j+k} = m_hat
    #   => t_j = (-1)^d_eff ( m_hat - sum_{k=1}^d_eff (-1)^{d_eff-k} C(d_eff,k) t_{j+k} )
    back_coeffs = np.array(
        [(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(1, d_eff + 1)],
        dtype=float,
    )
    coef0 = (-1) ** d_eff

    for j in range(start_tail - 1, -1, -1):
        # uses poly[j+1 .. j+d_eff], already filled
        sum_future = float(np.dot(back_coeffs, poly[j + 1:j + 1 + d_eff]))
        poly[j] = (m_hat - sum_future) * coef0

    # ----- forward recursion (t_i from t_{i-d_eff..i-1}) -----
    # coefficients for k=0..d_eff-1 in:
    #   t_i = m_hat - sum_{k=0}^{d_eff-1} (-1)^{d_eff-k} C(d_eff,k) t_{i-d_eff+k}
    fwd_coeffs = np.array(
        [(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(d_eff)],
        dtype=float,
    )

    for i in range(j0 + 1, N_total):
        sum_prev = float(np.dot(fwd_coeffs, poly[i - d_eff:i]))
        poly[i] = m_hat - sum_prev

    return poly


# ============================================================
#  Main
# ============================================================

def main():
    # 1. Load S&P 500
    t_all, Z_all, meta = load_sp500_series(
        ticker="NVDA",
        start="2010-01-01",
        end=None,
        use_log=True,
        csv_path=None,  # or path to CSV with Date / Adj Close
    )

    print(
        f"Loaded {meta['ticker']} from {meta['start']} to {meta['end']} "
        f"(N={Z_all.size}, log={meta['use_log']})."
    )

    # 2. Train / validation / test split
    Z_train, Z_val, Z_test, N_train, N_val, N_test = train_val_test_split(
        Z_all,
        frac_train=0.6,
        frac_val=0.2,
        min_train=80,
        min_val=60,
    )
    print(f"Train length = {N_train}, val = {N_val}, test = {N_test}.")

    # 3. Build spectral solvers per d (for train length)
    d_list = [1, 2, 3, 4]   # extend if numerically stable
    solvers: Dict[int, GuerreroSpectralSolver] = {}
    for d in d_list:
        print(f"\n=== Building spectral solver for d={d} ===")
        solvers[d] = GuerreroSpectralSolver(N_train, d)

    # 4. For each d, find ALL local minima in s
    all_minima_by_d: Dict[int, Dict[str, Any]] = {}
    for d in d_list:
        solver = solvers[d]

        def J_d(s: float, _solver=solver, _d=d) -> float:
            if N_val == 0:
                return np.nan
            try:
                t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = _solver.fit_for_s(Z_train, s)
            except Exception:
                return np.inf
            t_fore = forecast_trend(t_hat, d=_d, m_hat=m_hat, h=N_val)
            return float(np.mean((t_fore - Z_val) ** 2))

        print(f"\n=== Finding all local minima in s for d={d} ===")
        s_all, J_all = find_all_local_minima_s(
            J_d,
            d=d,
            s_min=1e-3,
            s_max=0.999,
            n_grid=400,       # grid resolution in s
            refine=True,
            refine_iter=25,
            verbose=True,
            plot=True,
        )
        all_minima_by_d[d] = dict(s_all=s_all, J_all=J_all)

    # 5. Visualizations per d with strict train/val/test, for ALL local minima
    for d in d_list:
        if d not in all_minima_by_d:
            print(f"\n--- Skipping d={d}: no minima stored ---")
            continue

        solver = solvers[d]
        s_all = all_minima_by_d[d]["s_all"]
        J_all = all_minima_by_d[d]["J_all"]

        for k, (s_star, J_star) in enumerate(zip(s_all, J_all), start=1):
            if not np.isfinite(s_star) or not np.isfinite(J_star):
                continue
            print(
                f"\n--- Plotting train/val/test for d={d}, "
                f"local minimum {k}/{len(s_all)}, "
                f"s≈{s_star:.4f}, val_MSE≈{J_star:.6e} ---"
            )
            plot_d_train_val_test_strict(
                Z_all=Z_all,
                N_train=N_train,
                N_val=N_val,
                d=d,
                s_unit=s_star,
                solver=solver,
                meta=meta,
            )


if __name__ == "__main__":
    main()
