# -*- coding: utf-8 -*-
"""
Guerrero (2007) penalized trend on S&P 500 with spectral solver.

For each differencing order d:

  - Define validation loss J_d(s) for smoothness index s∈(0,1).
  - Scan J_d(s) on a grid; detect ALL local minima in s.
  - Refine each local minimum with 1D golden-section.
  - For each local minimum, plot train/val/test fit.

"""

from typing import Callable, List, Tuple, Dict, Any, Optional
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
    save_pdf_path: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
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
        plt.figure()
        ax = plt.gca()
        ax.plot(s_grid, J_grid, label="J(s) (MSE de validación)", linewidth=1.5)
        ax.scatter(s_minima, J_minima, s=60, marker="*", color="k",
                   label="mínimos locales")
        ax.set_title(f"Mínimos locales de J(s) para d={d}")
        ax.set_xlabel("s")
        ax.set_ylabel("J(s) (MSE de validación)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        # save as pdf
        if save_pdf_path is not None and meta is not None:
            # create the path
            path_and_name = f"{save_pdf_path}/{meta['ticker']}/"
            import os 
            os.makedirs(os.path.dirname(path_and_name), exist_ok=True)
            plt.savefig(f"{path_and_name}minimaford_{d}.pdf")
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
    save_pdf_path: Optional[str] = None,
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
      - Annotate plot with RMSE and Weighted RMSE for
        train / val / train+val / test, using the same
        definitions as rmse_all_segments_for_s.
      - Save metrics to CSV {ticker}_d_{d}.csv:
        s, RMSE_train, RMSE_val, RMSE_both, RMSE_test,
        RMSEw_train, RMSEw_val, RMSEw_both, RMSEw_test.
    """
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    N_all = Z_all.size

    # --- split ---
    Z_train = Z_all[:N_train]
    Z_val   = Z_all[N_train:N_train + N_val]
    Z_test  = Z_all[N_train + N_val:]
    N_val   = Z_val.size
    N_test  = Z_test.size
    N_tv    = N_train + N_val  # train + val length

    # --- Guerrero fit on TRAIN ---
        # --- Guerrero fit on TRAIN (igual que antes) ---
    t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s_unit)
    t_hat = np.asarray(t_hat, dtype=float).ravel()

    # --- NUEVO: tendencia extendida por mínima rugosidad ---
    trend_full = extend_trend_minimum_roughness(
        t_hat_train=t_hat,
        d=d,
        N_total=N_all,
        N_train=N_train,
    )

    # --- indices for each segment ---
    idx_all   = np.arange(N_all)
    idx_train = idx_all[:N_train]
    idx_val   = idx_all[N_train:N_train + N_val]
    idx_test  = idx_all[N_train + N_val:]

    # --- segment predictions ---
    trend_train = trend_full[:N_train]
    trend_val   = trend_full[N_train:N_train + N_val]
    trend_test  = trend_full[N_train + N_val:]

    poly_train = trend_train
    poly_val   = trend_val
    poly_test  = trend_test

    # --- errors per segment (same as rmse_all_segments_for_s) ---
    err_train = poly_train - Z_train
    err_val   = poly_val   - Z_val
    err_test  = poly_test  - Z_test
    if N_tv > 0:
        err_tv = np.concatenate([err_train, err_val])
    else:
        err_tv = np.empty(0, dtype=float)

    # ============================
    # 1) Unweighted RMSEs (same formulas)
    # ============================
    rmse_train = float(np.sqrt(np.mean(err_train**2))) if N_train > 0 else np.nan
    rmse_val   = float(np.sqrt(np.mean(err_val**2)))   if N_val   > 0 else np.nan
    rmse_test  = float(np.sqrt(np.mean(err_test**2)))  if N_test  > 0 else np.nan
    rmse_both  = float(np.sqrt(np.mean(err_tv**2)))    if N_tv    > 0 else np.nan

    # ============================
    # 2) Weighted RMSEs (exactly as in rmse_all_segments_for_s)
    #    Requires w_train_seg, w_val_seg, w_test_seg, w_both in outer scope
    # ============================
    if N_val == 0:
        raise ValueError("Validation set is empty; cannot define J_val(s).")

    # ----- time weights per segment (later times heavier in each segment) -----
    if N_train > 0:
        idx_tr_w = np.arange(N_train, dtype=float)
        w_train_seg = (idx_tr_w + 1.0) / np.sum(idx_tr_w + 1.0)
    else:
        w_train_seg = np.empty(0, dtype=float)

    if N_val > 0:
        idx_val_w = np.arange(N_val, dtype=float)
        w_val_seg = (idx_val_w + 1.0) / np.sum(idx_val_w + 1.0)
    else:
        w_val_seg = np.empty(0, dtype=float)

    if N_test > 0:
        idx_te_w = np.arange(N_test, dtype=float)
        w_test_seg = (idx_te_w + 1.0) / np.sum(idx_te_w + 1.0)
    else:
        w_test_seg = np.empty(0, dtype=float)
    
    if N_tv > 0:
        idx_tv_w = np.arange(N_tv, dtype=float)
        w_both = (idx_tv_w + 1.0) / np.sum(idx_tv_w + 1.0)
    else:
        w_both = np.empty(0, dtype=float)

    # train
    if N_train > 0:
        rmse_w_train = float(np.sqrt(np.sum(w_train_seg * (err_train ** 2))))
    else:
        rmse_w_train = np.nan

    # val
    if N_val > 0:
        rmse_w_val = float(np.sqrt(np.sum(w_val_seg * (err_val ** 2))))
    else:
        rmse_w_val = np.nan

    # test
    if N_test > 0:
        rmse_w_test = float(np.sqrt(np.sum(w_test_seg * (err_test ** 2))))
    else:
        rmse_w_test = np.nan

    # train + val
    if N_tv > 0:
        rmse_w_both = float(np.sqrt(np.sum(w_both * (err_tv ** 2))))
    else:
        rmse_w_both = np.nan

    # Optional: J_val = MSE on validation (no sqrt), using same polynomial
    J_val = float(np.mean((poly_val - Z_val) ** 2)) if N_val > 0 else np.nan

    # --- Plot ---
    plt.figure()
    ax = plt.gca()

    # observed series with color
    alpha = 1
    ax.plot(idx_train, Z_train, label=r"$Z_t$ (entrenamiento)", linewidth=0.5, color="#03045e", alpha=alpha)
    if N_val > 0:
        ax.plot(idx_val, Z_val, label=r"$Z_t$ (validación)", linewidth=0.5, color="#023e8a", alpha=alpha)
    if N_test > 0:
        ax.plot(idx_test, Z_test, label=r"$Z_t$ (prueba)", linewidth=0.5, color="#0077b6", alpha=alpha)
    # Guerrero trend on train
        # Tendencia ajustada en TRAIN (Guerrero puro, sólo en train)
    ax.plot(
        idx_train,
        t_hat,
        label=r"Tendencia (ajustada en entrenamiento)",
        linewidth=2.0,
        color="#d00000",
    )

    # Tendencia extendida por mínima rugosidad sobre todo el horizonte
    ax.plot(
        idx_all,
        trend_full,
        linestyle="--",
        linewidth=1.8,
        color="#ffba08",
        label=r"Tendencia extendida (mínima rugosidad)",
    )


    # vertical split markers
    ax.axvline(N_train - 0.5, color="k", linestyle="--", linewidth=1)
    if N_val > 0:
        ax.axvline(N_train + N_val - 0.5, color="k", linestyle="--", linewidth=1)

    # ============================
    # Title with same style / naming as your verbose print
    # ============================

    # choose which s to display (here: s_real if available, else s_unit)
    s_disp = s_real if not np.isnan(s_real) else s_unit

    base_line = (
        f"{meta['ticker']} – "
        f"d={d}, s≈{s_disp:.6f}, λ≈{lam:.3g}, m̂≈{m_hat:.4f}"
    )
    if not np.isnan(J_val):
        base_line += f", J_val≈{J_val:.6e}"

    # Unweighted RMSEs
    title_line2 = (
        f"RMSEtr = {rmse_train:.4f}, "
        f"RMSEvl = {rmse_val:.4f}, "
        f"RMSEts = {rmse_test:.4f}, "
        f"RMSEtv = {rmse_both:.4f}"
    )

    # Weighted RMSEs
    title_line3 = (
        f"RMSEWtr = {rmse_w_train:.4f}, "
        f"RMSEWvl = {rmse_w_val:.4f}, "
        f"RMSEWts = {rmse_w_test:.4f}, "
        f"RMSEWtv = {rmse_w_both:.4f}"
    )

    ax.set_title(base_line + "\n" + title_line2 + "\n" + title_line3)

    ax.set_xlabel("Días")
    ax.set_ylabel("Log-precio" if meta.get("use_log", False) else "Precio")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    # save as pdf
    if save_pdf_path is not None:
        plt.savefig(f"{save_pdf_path}_d_{d}_s_{s_unit:.3f}.pdf")

    plt.show()

    # ============================
    # Save metrics to CSV (one row per s)
    # ============================
    import os
    import csv

    csv_dir = "results_csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{meta['ticker']}_d_{d}.csv")

    fieldnames = [
        "s",
        "RMSE_train",
        "RMSE_val",
        "RMSE_both",
        "RMSE_test",
        "RMSEw_train",
        "RMSEw_val",
        "RMSEw_both",
        "RMSEw_test",
    ]

    row = {
        "s":          float(s_disp),
        "RMSE_train": rmse_train,
        "RMSE_val":   rmse_val,
        "RMSE_both":  rmse_both,
        "RMSE_test":  rmse_test,
        "RMSEw_train": rmse_w_train,
        "RMSEw_val":   rmse_w_val,
        "RMSEw_both":  rmse_w_both,
        "RMSEw_test":  rmse_w_test,
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)




def extend_trend_minimum_roughness(
    t_hat_train: np.ndarray,
    d: int,
    N_total: int,
    N_train: int,
) -> np.ndarray:
    """
    Extiende la tendencia ajustada en TRAIN a todo el horizonte [0..N_total-1]
    SIN imponer Δ^d t_t = m_hat constante.

    Idea:
      - Fijamos los primeros N_train puntos en t_hat_train.
      - Elegimos los N_future = N_total - N_train puntos futuros para
        minimizar la rugosidad global:
            min_u || K_full t_full ||^2,
        donde:
          * K_full = matriz de diferencias de orden d sobre [0..N_total-1]
          * t_full = [t_hat_train; u]

        Esto sólo penaliza Δ^d t_t, no usa m_hat ni observa Z en val/test.
        Es la “extensión más suave” compatible con la cola del train.

    Notas:
      - No entra λ: si sólo hay término de penalización, el factor de escala
        se cancela en el argmin.
      - No hay data leakage: para t >= N_train no se usa Z_t.

    Parámetros
    ----------
    t_hat_train : tendencia en TRAIN (longitud N_train)
    d           : orden de diferencia (1, 2, 3, 4, ...)
    N_total     : longitud total (TRAIN + VAL + TEST)
    N_train     : longitud del tramo de entrenamiento

    Devuelve
    --------
    t_full : np.ndarray
        Tendencia extendida de longitud N_total.
    """
    t_hat_train = np.asarray(t_hat_train, dtype=float).ravel()
    N_train = int(N_train)
    N_total = int(N_total)

    if N_total <= N_train:
        # No hay nada que extender
        return t_hat_train.copy()

    N_future = N_total - N_train
    d_eff = int(d)

    # Caso trivial d=0: mantenemos constante la tendencia
    if d_eff <= 0:
        t_full = np.empty(N_total, dtype=float)
        t_full[:N_train] = t_hat_train
        t_full[N_train:] = t_hat_train[-1]
        return t_full

    # Matriz de diferencias sobre TODO el horizonte (0..N_total-1)
    K_full = difference_matrix(N_total, d_eff)  # shape: (N_total - d_eff, N_total)

    # Particionamos columnas: conocidas (TRAIN) y desconocidas (FUTURO)
    K_k = K_full[:, :N_train]       # columnas 0 .. N_train-1
    K_u = K_full[:, N_train:]       # columnas N_train .. N_total-1

    # Queremos u minimizando || K_k k + K_u u ||^2
    # Normal equations: (K_u' K_u) u = - K_u' K_k k
    k_vec = t_hat_train
    A = K_u.T @ K_u          # (N_future x N_future)
    b = -K_u.T @ (K_k @ k_vec)

    # Resolver sistema; usamos lstsq por robustez numérica
    u, *_ = np.linalg.lstsq(A, b, rcond=None)

    t_full = np.empty(N_total, dtype=float)
    t_full[:N_train] = k_vec
    t_full[N_train:] = u
    return t_full







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



def analyze_all_objectives_for_d(
    solver: GuerreroSpectralSolver,
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    Z_test: np.ndarray,
    d: int,
    s_min: float = 1e-3,
    s_max: float = 0.999,
    n_grid: int = 250,
    refine: bool = True,
    refine_iter: int = 20,
    verbose: bool = True,
    plot: bool = True,
    save_pdf_path: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For a fixed d, compute five objectives as functions of s:

        J_train(s)  : MSE on TRAIN (vs Δ^d polynomial)
        J_val(s)    : unweighted MSE on VAL
        J_both(s)   : unweighted MSE on TRAIN+VAL (length-weighted)
        J_val_w(s)  : time-weighted MSE on VAL (latest VAL points heavier)
        J_both_w(s) : time-weighted MSE on TRAIN+VAL (later times heavier)

    Steps:
      1) Evaluate all five on a grid of n_grid points in [s_min, s_max].
      2) Detect all discrete local minima for each objective.
      3) Optionally refine each minimum by local golden-section.

    For every minimum we also print:
      s, J_*,
      RMSE_train, RMSE_val, RMSE_test,
      RMSE_train_and_val,
      RMSE_Weighted_train, RMSE_Weighted_val, RMSE_Weighted_test,
      RMSE_Weighted_train_and_val,

    where all RMSEs are w.r.t. the global Δ^d polynomial (train+val+test).
    """
    Z_train = np.asarray(Z_train, dtype=float).ravel()
    Z_val = np.asarray(Z_val, dtype=float).ravel()
    Z_test = np.asarray(Z_test, dtype=float).ravel()
    N_train = Z_train.size
    N_val = Z_val.size
    N_test = Z_test.size
    N_tv = N_train + N_val          # train+val length
    N_full = N_train + N_val + N_test

    if N_val == 0:
        raise ValueError("Validation set is empty; cannot define J_val(s).")

    # ----- time weights per segment (later times heavier in each segment) -----
    if N_train > 0:
        idx_tr = np.arange(N_train, dtype=float)
        w_train_seg = (idx_tr + 1.0) / np.sum(idx_tr + 1.0)
    else:
        w_train_seg = np.empty(0, dtype=float)

    if N_val > 0:
        idx_val = np.arange(N_val, dtype=float)
        w_val_seg = (idx_val + 1.0) / np.sum(idx_val + 1.0)
    else:
        w_val_seg = np.empty(0, dtype=float)

    if N_test > 0:
        idx_te = np.arange(N_test, dtype=float)
        w_test_seg = (idx_te + 1.0) / np.sum(idx_te + 1.0)
    else:
        w_test_seg = np.empty(0, dtype=float)

    # ----- time weights for TRAIN+VAL together (later overall times heavier) -----
    if N_tv > 0:
        idx_tv = np.arange(N_tv, dtype=float)
        w_both = (idx_tv + 1.0) / np.sum(idx_tv + 1.0)
    else:
        w_both = np.empty(0, dtype=float)

    # ---------- 1) evaluate on a grid ----------
    s_grid = np.linspace(s_min, s_max, n_grid)
    J_train_grid = np.empty_like(s_grid)
    J_val_grid   = np.empty_like(s_grid)
    J_both_grid  = np.empty_like(s_grid)
    J_val_w_grid = np.empty_like(s_grid)
    J_both_w_grid = np.empty_like(s_grid)

    for i, s in enumerate(s_grid):
        try:
            t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s)
        except Exception:
            J_train_grid[i] = np.inf
            J_val_grid[i]   = np.inf
            J_both_grid[i]  = np.inf
            J_val_w_grid[i] = np.inf
            J_both_w_grid[i] = np.inf
            continue

        # polynomial on TRAIN+VAL only (no need for TEST here)
        poly_tv = build_polynomial_from_train_tail(
            t_hat_train=t_hat,
            d=d,
            m_hat=m_hat,
            N_total=N_tv,
            N_train=N_train,
        )
        poly_train = poly_tv[:N_train]
        poly_val   = poly_tv[N_train:]

        err_train = poly_train - Z_train
        err_val   = poly_val   - Z_val
        err_tv    = np.concatenate([err_train, err_val])

        # unweighted objectives
        Jtr = float(np.mean(err_train ** 2))
        Jva = float(np.mean(err_val   ** 2))
        Jbo = (N_train * Jtr + N_val * Jva) / N_tv

        # weighted objectives
        if N_val > 0:
            Jva_w = float(np.sum(w_val_seg * (err_val ** 2)))
        else:
            Jva_w = np.inf
        if N_tv > 0:
            Jbo_w = float(np.sum(w_both * (err_tv ** 2)))
        else:
            Jbo_w = np.inf

        J_train_grid[i] = Jtr
        J_val_grid[i]   = Jva
        J_both_grid[i]  = Jbo
        J_val_w_grid[i] = Jva_w
        J_both_w_grid[i] = Jbo_w

    # ---------- helper: detect + refine local minima ----------
    def detect_minima(J_grid: np.ndarray, J_fun: Callable[[float], float]):
        J_cmp = np.array(J_grid)
        J_cmp[~np.isfinite(J_cmp)] = np.inf

        idx_cand = []
        # interior points
        for idx in range(1, n_grid - 1):
            if not np.isfinite(J_cmp[idx]):
                continue
            if (J_cmp[idx] <= J_cmp[idx - 1]) and (J_cmp[idx] <= J_cmp[idx + 1]):
                idx_cand.append(idx)
        # endpoints
        if np.isfinite(J_cmp[0]) and J_cmp[0] <= J_cmp[1]:
            idx_cand.append(0)
        if np.isfinite(J_cmp[-1]) and J_cmp[-1] <= J_cmp[-2]:
            idx_cand.append(n_grid - 1)

        s_list, J_list = [], []
        for idx in idx_cand:
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
                s_star, J_star = golden_local(J_fun, a, b, n_iter=refine_iter)
            else:
                s_star, J_star = s0, J_grid[idx]

            s_list.append(s_star)
            J_list.append(J_star)

        s_arr = np.array(s_list, dtype=float)
        J_arr = np.array(J_list, dtype=float)
        order = np.argsort(s_arr)
        return s_arr[order], J_arr[order]

    # ---------- pointwise J(s) for refinement ----------
    def J_train_fun(s: float) -> float:
        try:
            t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s)
        except Exception:
            return np.inf
        poly_full = build_polynomial_from_train_tail(
            t_hat_train=t_hat,
            d=d,
            m_hat=m_hat,
            N_total=N_full,
            N_train=N_train,
        )
        poly_train = poly_full[:N_train]
        return float(np.mean((poly_train - Z_train) ** 2))

    def J_val_fun(s: float) -> float:
        try:
            t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s)
        except Exception:
            return np.inf
        poly_full = build_polynomial_from_train_tail(
            t_hat_train=t_hat,
            d=d,
            m_hat=m_hat,
            N_total=N_full,
            N_train=N_train,
        )
        poly_val = poly_full[N_train:N_train + N_val]
        return float(np.mean((poly_val - Z_val) ** 2))

    def J_both_fun(s: float) -> float:
        try:
            t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s)
        except Exception:
            return np.inf
        poly_full = build_polynomial_from_train_tail(
            t_hat_train=t_hat,
            d=d,
            m_hat=m_hat,
            N_total=N_full,
            N_train=N_train,
        )
        poly_train = poly_full[:N_train]
        poly_val   = poly_full[N_train:N_train + N_val]
        Jtr = float(np.mean((poly_train - Z_train) ** 2))
        Jva = float(np.mean((poly_val   - Z_val)   ** 2))
        return (N_train * Jtr + N_val * Jva) / (N_train + N_val)

    def J_val_w_fun(s: float) -> float:
        try:
            t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s)
        except Exception:
            return np.inf
        poly_full = build_polynomial_from_train_tail(
            t_hat_train=t_hat,
            d=d,
            m_hat=m_hat,
            N_total=N_full,
            N_train=N_train,
        )
        poly_val = poly_full[N_train:N_train + N_val]
        err_val = poly_val - Z_val
        if N_val == 0:
            return np.inf
        return float(np.sum(w_val_seg * (err_val ** 2)))

    def J_both_w_fun(s: float) -> float:
        try:
            t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s)
        except Exception:
            return np.inf
        poly_full = build_polynomial_from_train_tail(
            t_hat_train=t_hat,
            d=d,
            m_hat=m_hat,
            N_total=N_full,
            N_train=N_train,
        )
        poly_train = poly_full[:N_train]
        poly_val   = poly_full[N_train:N_train + N_val]
        err_train  = poly_train - Z_train
        err_val    = poly_val   - Z_val
        err_tv     = np.concatenate([err_train, err_val])
        if N_tv == 0:
            return np.inf
        return float(np.sum(w_both * (err_tv ** 2)))

    # helper: RMSEs (train/val/test, and train+val; unweighted and weighted)
    def rmse_all_segments_for_s(s: float):
        try:
            t_hat, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(Z_train, s)
        except Exception:
            return (np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan)

                # === NUEVO: extensión por mínima rugosidad, sin imponer Δ^d t_t = const ===
        trend_full = extend_trend_minimum_roughness(
            t_hat_train=t_hat,
            d=d,
            N_total=N_full,
            N_train=N_train,
        )
        trend_train = trend_full[:N_train]
        trend_val   = trend_full[N_train:N_train + N_val]
        trend_test  = trend_full[N_train + N_val:]

        err_train = trend_train - Z_train
        err_val   = trend_val   - Z_val
        err_test  = trend_test  - Z_test
        err_tv    = np.concatenate([err_train, err_val]) if N_tv > 0 else np.empty(0)


        # unweighted RMSE
        rmse_tr = float(np.sqrt(np.mean(err_train ** 2))) if N_train > 0 else np.nan
        rmse_va = float(np.sqrt(np.mean(err_val   ** 2))) if N_val   > 0 else np.nan
        rmse_te = float(np.sqrt(np.mean(err_test  ** 2))) if N_test  > 0 else np.nan
        rmse_tv = float(np.sqrt(np.mean(err_tv    ** 2))) if N_tv    > 0 else np.nan

        # weighted RMSE per segment
        if N_train > 0:
            rmse_tr_w = float(np.sqrt(np.sum(w_train_seg * (err_train ** 2))))
        else:
            rmse_tr_w = np.nan
        if N_val > 0:
            rmse_va_w = float(np.sqrt(np.sum(w_val_seg * (err_val ** 2))))
        else:
            rmse_va_w = np.nan
        if N_test > 0:
            rmse_te_w = float(np.sqrt(np.sum(w_test_seg * (err_test ** 2))))
        else:
            rmse_te_w = np.nan

        if N_tv > 0:
            rmse_tv_w = float(np.sqrt(np.sum(w_both * (err_tv ** 2))))
        else:
            rmse_tv_w = np.nan

        return (rmse_tr, rmse_va, rmse_te,
                rmse_tr_w, rmse_va_w, rmse_te_w,
                rmse_tv, rmse_tv_w)

    # ---------- 2) detect + refine for each objective ----------
    s_tr, J_tr = detect_minima(J_train_grid, J_train_fun)
    s_va, J_va = detect_minima(J_val_grid,   J_val_fun)
    s_bo, J_bo = detect_minima(J_both_grid,  J_both_fun)
    s_vw, J_vw = detect_minima(J_val_w_grid, J_val_w_fun)
    s_bw, J_bw = detect_minima(J_both_w_grid, J_both_w_fun)

    # ---------- 2b) verbose prints with full RMSE info ----------
    if verbose:
        print(f"\n[d={d}] TRAIN minima:")
        for s_star, J_star in zip(s_tr, J_tr):
            (rmse_tr, rmse_va, rmse_te,
             rmse_tr_w, rmse_va_w, rmse_te_w,
             rmse_tv, rmse_tv_w) = rmse_all_segments_for_s(s_star)
            print(
                f"  s≈{s_star:.6f}, J_train≈{J_star:.6e}, "
                f"RMSE_train≈{rmse_tr:.4f}, RMSE_val≈{rmse_va:.4f}, RMSE_test≈{rmse_te:.4f}, "
                f"RMSE_train_and_val≈{rmse_tv:.4f}, "
                f"RMSE_Weighted_train≈{rmse_tr_w:.4f}, RMSE_Weighted_val≈{rmse_va_w:.4f}, "
                f"RMSE_Weighted_test≈{rmse_te_w:.4f}, "
                f"RMSE_Weighted_train_and_val≈{rmse_tv_w:.4f}"
            )

        print(f"[d={d}] VAL minima:")
        for s_star, J_star in zip(s_va, J_va):
            (rmse_tr, rmse_va, rmse_te,
             rmse_tr_w, rmse_va_w, rmse_te_w,
             rmse_tv, rmse_tv_w) = rmse_all_segments_for_s(s_star)
            print(
                f"  s≈{s_star:.6f}, J_val≈{J_star:.6e}, "
                f"RMSE_train≈{rmse_tr:.4f}, RMSE_val≈{rmse_va:.4f}, RMSE_test≈{rmse_te:.4f}, "
                f"RMSE_train_and_val≈{rmse_tv:.4f}, "
                f"RMSE_Weighted_train≈{rmse_tr_w:.4f}, RMSE_Weighted_val≈{rmse_va_w:.4f}, "
                f"RMSE_Weighted_test≈{rmse_te_w:.4f}, "
                f"RMSE_Weighted_train_and_val≈{rmse_tv_w:.4f}"
            )

        print(f"[d={d}] BOTH minima:")
        for s_star, J_star in zip(s_bo, J_bo):
            (rmse_tr, rmse_va, rmse_te,
             rmse_tr_w, rmse_va_w, rmse_te_w,
             rmse_tv, rmse_tv_w) = rmse_all_segments_for_s(s_star)
            print(
                f"  s≈{s_star:.6f}, J_both≈{J_star:.6e}, "
                f"RMSE_train≈{rmse_tr:.4f}, RMSE_val≈{rmse_va:.4f}, RMSE_test≈{rmse_te:.4f}, "
                f"RMSE_train_and_val≈{rmse_tv:.4f}, "
                f"RMSE_Weighted_train≈{rmse_tr_w:.4f}, RMSE_Weighted_val≈{rmse_va_w:.4f}, "
                f"RMSE_Weighted_test≈{rmse_te_w:.4f}, "
                f"RMSE_Weighted_train_and_val≈{rmse_tv_w:.4f}"
            )

        print(f"[d={d}] VAL-WEIGHTED minima:")
        for s_star, J_star in zip(s_vw, J_vw):
            (rmse_tr, rmse_va, rmse_te,
             rmse_tr_w, rmse_va_w, rmse_te_w,
             rmse_tv, rmse_tv_w) = rmse_all_segments_for_s(s_star)
            print(
                f"  s≈{s_star:.6f}, J_val_w≈{J_star:.6e}, "
                f"RMSE_train≈{rmse_tr:.4f}, RMSE_val≈{rmse_va:.4f}, RMSE_test≈{rmse_te:.4f}, "
                f"RMSE_train_and_val≈{rmse_tv:.4f}, "
                f"RMSE_Weighted_train≈{rmse_tr_w:.4f}, RMSE_Weighted_val≈{rmse_va_w:.4f}, "
                f"RMSE_Weighted_test≈{rmse_te_w:.4f}, "
                f"RMSE_Weighted_train_and_val≈{rmse_tv_w:.4f}"
            )

        print(f"[d={d}] BOTH-WEIGHTED minima:")
        for s_star, J_star in zip(s_bw, J_bw):
            (rmse_tr, rmse_va, rmse_te,
             rmse_tr_w, rmse_va_w, rmse_te_w,
             rmse_tv, rmse_tv_w) = rmse_all_segments_for_s(s_star)
            print(
                f"  s≈{s_star:.6f}, J_both_w≈{J_star:.6e}, "
                f"RMSE_train≈{rmse_tr:.4f}, RMSE_val≈{rmse_va:.4f}, RMSE_test≈{rmse_te:.4f}, "
                f"RMSE_train_and_val≈{rmse_tv:.4f}, "
                f"RMSE_Weighted_train≈{rmse_tr_w:.4f}, RMSE_Weighted_val≈{rmse_va_w:.4f}, "
                f"RMSE_Weighted_test≈{rmse_te_w:.4f}, "
                f"RMSE_Weighted_train_and_val≈{rmse_tv_w:.4f}"
            )

    # ---------- 3) plot ----------
    if plot:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

        ax.plot(s_grid, J_val_grid,   label="J_val(s)",   linewidth=1.8)
        ax.plot(s_grid, J_train_grid, label="J_train(s)", linewidth=1.0)
        ax.plot(s_grid, J_both_grid,  label="J_both(s)",  linewidth=1.0, linestyle="--")
        ax.plot(s_grid, J_val_w_grid, label="J_val_w(s) (val weighted)", linewidth=1.3, linestyle="-.")
        ax.plot(s_grid, J_both_w_grid, label="J_both_w(s) (train+val weighted)", linewidth=1.3, linestyle=":")

        if len(s_va) > 0:
            ax.scatter(s_va, J_va, s=60, marker="o", color="C0", label="val minima")
        if len(s_tr) > 0:
            ax.scatter(s_tr, J_tr, s=60, marker="s", color="C1", label="train minima")
        if len(s_bo) > 0:
            ax.scatter(s_bo, J_bo, s=70, marker="*", color="C2", label="both minima")
        if len(s_vw) > 0:
            ax.scatter(s_vw, J_vw, s=60, marker="^", color="C3", label="val_w minima")
        if len(s_bw) > 0:
            ax.scatter(s_bw, J_bw, s=60, marker="v", color="C4", label="both_w minima")

        ax.set_title(f"Local minima of J_train, J_val, J_both, J_val_w, J_both_w for d={d}")
        ax.set_xlabel("s")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        # save as pdf
        if save_pdf_path is not None:
            plt.savefig(f"{save_pdf_path}_d_{d}_objectives.pdf")
        plt.show()

    return {
        "train": {"s": s_tr, "J": J_tr},
        "val":   {"s": s_va, "J": J_va},
        "both":  {"s": s_bo, "J": J_bo},
        "val_w": {"s": s_vw, "J": J_vw},
        "both_w": {"s": s_bw, "J": J_bw},
        "s_grid": s_grid,
        "J_train_grid": J_train_grid,
        "J_val_grid": J_val_grid,
        "J_both_grid": J_both_grid,
        "J_val_w_grid": J_val_w_grid,
        "J_both_w_grid": J_both_w_grid,
    }





# ============================================================
#  Main
# ============================================================

def main(ticker: str = "NVDA", start: str = "2010-01-01", save_pdf_path: Optional[str] = None, d_list = [1, 2, 3]):
    # 1. Load S&P 500
    t_all, Z_all, meta = load_sp500_series(
        ticker=ticker,
        start=start,
        end=None,
        use_log=True,
        csv_path=None,  # or path to CSV with Date / Adj Close
    )

    if save_pdf_path is not None:
        # create the path
        path_and_name = f"{save_pdf_path}/{meta['ticker']}/"
        import os 
        os.makedirs(os.path.dirname(path_and_name), exist_ok=True)
        print(f"PDFs will be saved to: {path_and_name}")

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
    solvers: Dict[int, GuerreroSpectralSolver] = {}
    for d in d_list:
        print(f"\n=== Building spectral solver for d={d} ===")
        solvers[d] = GuerreroSpectralSolver(N_train, d)

    # 4. For each d, find ALL local minima in s
        # 4. For each d, analyze ALL objectives and their local minima
    all_minima_by_d: Dict[int, Dict[str, Any]] = {}
    for d in d_list:
        solver = solvers[d]

        print(f"\n=== Analyzing objectives & local minima for d={d} ===")
        res = analyze_all_objectives_for_d(
            solver=solver,
            Z_train=Z_train,
            Z_val=Z_val,
            Z_test=Z_test,
            d=d,
            s_min=1e-5,
            s_max=0.9999,
            n_grid=500,
            refine=True,
            refine_iter=20,
            verbose=True,
            plot=True,
            save_pdf_path=path_and_name if save_pdf_path is not None else None,
        )

        all_minima_by_d[d] = res


    # 5. Visualizations per d with strict train/val/test, for ALL minima
    for d in d_list:
        solver = solvers[d]
        res = all_minima_by_d[d]

        # order of objectives: val, val_w, both, both_w, train
        for obj_name in ["val", "val_w", "both", "both_w", "train"]:
            s_all = res[obj_name]["s"]
            J_all = res[obj_name]["J"]
            for k, (s_star, J_star) in enumerate(zip(s_all, J_all), start=1):
                if not np.isfinite(s_star) or not np.isfinite(J_star):
                    continue
                print(
                    f"\n--- Plotting train/val/test for d={d}, "
                    f"{obj_name} minimum {k}/{len(s_all)}, "
                    f"s≈{s_star:.4f}, MSE≈{J_star:.6e} ---"
                )
                plot_d_train_val_test_strict(
                    Z_all=Z_all,
                    N_train=N_train,
                    N_val=N_val,
                    d=d,
                    s_unit=s_star,
                    solver=solver,
                    meta=meta,
                    save_pdf_path=path_and_name if save_pdf_path is not None else None,
                )

    


if __name__ == "__main__":
    main()
