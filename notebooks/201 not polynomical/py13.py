# -*- coding: utf-8 -*-
"""
py13_fullseries_smoother.py

Ex-post Guerrero-type full-series smoother (non-polynomial tail).

This module is meant ONLY for descriptive smoothing of the entire
series (train + val + test) after you have already selected a
smoothness index s* using py12_timeweighted.py on a proper
train/validation scheme.

Important:
    - Here, the Guerrero smoother is fit on the *full* observed
      series Z_all (train+val+test).
    - Therefore, residuals on "validation" and "test" segments are
      IN-SAMPLE diagnostics, NOT out-of-sample errors.
    - For proper model selection and test RMSE, use py12_timeweighted.py.
"""

import os
import sys
import importlib

# Add the path to the val03 folder to sys.path so modules in the same folder can be imported
sys.path.append(os.path.join('notebooks', '201 not polynomical'))

from typing import Dict, Any, Optional, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt

from py12_timeweighted import (
    GuerreroSpectralSolver,
    load_sp500_series,
    train_val_test_split,
)


# ============================================================
#  Full-series smoothing and diagnostics
# ============================================================

def compute_time_weights_segment(N: int) -> np.ndarray:
    """
    Linearly increasing time-weights within a segment of length N.
    Later points get higher weight.

    Returns a 1D array w of length N with sum(w) = 1.
    """
    if N <= 0:
        return np.empty(0, dtype=float)
    idx = np.arange(N, dtype=float)
    w = (idx + 1.0) / np.sum(idx + 1.0)
    return w


def smooth_full_series_for_d(
    Z_all: np.ndarray,
    d: int,
    s_unit: float,
    m_tol: float = 1e-10,
    max_m_iter: int = 120,
) -> Tuple[np.ndarray, float, float, float, np.ndarray, float]:
    """
    Fit Guerrero smoother of order d on the *full* series Z_all
    at a given smoothness index s_unit in (0, 1).

    This is a standard Guerrero fit with N = len(Z_all); it does not
    know about train/val/test splits.

    Parameters
    ----------
    Z_all : array-like, shape (N,)
        Full series (train+val+test concatenated).
    d : int
        Differencing order used in the penalty.
    s_unit : float
        Target normalized smoothness index in (0, 1).
    m_tol : float
        Tolerance for the fixed-point iteration in m.
    max_m_iter : int
        Max iterations for the fixed-point iteration.

    Returns
    -------
    t_hat_full : np.ndarray, shape (N,)
        Smoothed trend over the *entire* sample.
    m_hat : float
        Estimated drift of Δ^d t_t on the full sample.
    lam : float
        Smoothing parameter λ corresponding to s_unit.
    sigma2_hat : float
        Variance estimate from the Guerrero fit.
    diag_Ainv : np.ndarray, shape (N,)
        Diagonal of (I + λ KᵀK)⁻¹.
    s_real : float
        Realized normalized smoothness (may differ slightly from s_unit).
    """
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    N_all = Z_all.size

    solver = GuerreroSpectralSolver(N_all, d)
    t_hat_full, m_hat, lam, sigma2_hat, diag_Ainv, s_real = solver.fit_for_s(
        Z_all,
        s_unit,
        m_tol=m_tol,
        max_m_iter=max_m_iter,
    )
    t_hat_full = np.asarray(t_hat_full, dtype=float).ravel()
    if t_hat_full.size != N_all:
        raise ValueError(
            f"Expected t_hat of length {N_all}, got {t_hat_full.size}."
        )
    return t_hat_full, m_hat, lam, sigma2_hat, diag_Ainv, s_real


def plot_fullseries_split_nonpoly(
    Z_all: np.ndarray,
    N_train: int,
    N_val: int,
    d: int,
    s_unit: float,
    meta: Dict[str, Any],
    save_pdf_path: Optional[str] = None,
    m_tol: float = 1e-10,
    max_m_iter: int = 120,
):
    """
    Plot full-series Guerrero smoother (non-polynomial tail) with
    train / val / test segments indicated.

    The smoother is fit on Z_all (train+val+test). Residuals on each
    segment are IN-SAMPLE diagnostics, not out-of-sample performance.

    Parameters
    ----------
    Z_all : array-like
        Full series (train+val+test concatenated).
    N_train : int
        Length of the training segment.
    N_val : int
        Length of the validation segment.
    d : int
        Differencing order.
    s_unit : float
        Smoothness index chosen previously (e.g., from py12_timeweighted).
    meta : dict
        Metadata dict from load_sp500_series (ticker, use_log, etc.).
    save_pdf_path : str or None
        Base directory to save PDFs. If None, no file is saved.
    m_tol, max_m_iter : float, int
        Parameters forwarded to the Guerrero solver.
    """
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    N_all = Z_all.size

    Z_train = Z_all[:N_train]
    Z_val = Z_all[N_train:N_train + N_val]
    Z_test = Z_all[N_train + N_val:]

    N_val = Z_val.size
    N_test = Z_test.size
    N_tv = N_train + N_val

    # Fit full-series smoother (non-polynomial extension)
    t_hat_full, m_hat, lam, sigma2_hat, diag_Ainv, s_real = smooth_full_series_for_d(
        Z_all=Z_all,
        d=d,
        s_unit=s_unit,
        m_tol=m_tol,
        max_m_iter=max_m_iter,
    )

    idx_all = np.arange(N_all)
    idx_train = idx_all[:N_train]
    idx_val = idx_all[N_train:N_train + N_val]
    idx_test = idx_all[N_train + N_val:]

    # In-sample residuals per segment
    err_train = t_hat_full[idx_train] - Z_train
    err_val = t_hat_full[idx_val] - Z_val
    err_test = t_hat_full[idx_test] - Z_test
    err_tv = np.concatenate([err_train, err_val]) if N_tv > 0 else np.empty(0, dtype=float)

    # Unweighted IN-SAMPLE RMSEs
    rmse_in_train = float(np.sqrt(np.mean(err_train ** 2))) if N_train > 0 else np.nan
    rmse_in_val = float(np.sqrt(np.mean(err_val ** 2))) if N_val > 0 else np.nan
    rmse_in_test = float(np.sqrt(np.mean(err_test ** 2))) if N_test > 0 else np.nan
    rmse_in_both = float(np.sqrt(np.mean(err_tv ** 2))) if N_tv > 0 else np.nan

    # Time-weights within segments for weighted IN-SAMPLE RMSE
    w_train = compute_time_weights_segment(N_train)
    w_val = compute_time_weights_segment(N_val)
    w_test = compute_time_weights_segment(N_test)
    w_both = compute_time_weights_segment(N_tv)

    if N_train > 0:
        rmsew_in_train = float(np.sqrt(np.sum(w_train * (err_train ** 2))))
    else:
        rmsew_in_train = np.nan
    if N_val > 0:
        rmsew_in_val = float(np.sqrt(np.sum(w_val * (err_val ** 2))))
    else:
        rmsew_in_val = np.nan
    if N_test > 0:
        rmsew_in_test = float(np.sqrt(np.sum(w_test * (err_test ** 2))))
    else:
        rmsew_in_test = np.nan
    if N_tv > 0:
        rmsew_in_both = float(np.sqrt(np.sum(w_both * (err_tv ** 2))))
    else:
        rmsew_in_both = np.nan

    # ----------------- Plot -----------------
    plt.figure()
    ax = plt.gca()

    # Observed series
    ax.plot(idx_train, Z_train, label=r"$Z_t$ (entrenamiento)", linewidth=1.0)
    if N_val > 0:
        ax.plot(idx_val, Z_val, label=r"$Z_t$ (validación)", linewidth=1.0)
    if N_test > 0:
        ax.plot(idx_test, Z_test, label=r"$Z_t$ (prueba)", linewidth=1.0)

    # Full-series smoothed trend (non-polynomial tail)
    ax.plot(
        idx_all,
        t_hat_full,
        linestyle="--",
        linewidth=2.0,
        label=r"Tendencia suavizada (serie completa)",
    )

    # Vertical split markers
    ax.axvline(N_train - 0.5, color="k", linestyle="--", linewidth=1)
    if N_val > 0:
        ax.axvline(N_train + N_val - 0.5, color="k", linestyle="--", linewidth=1)

    # Title: emphasize IN-SAMPLE nature of the errors
    s_disp = s_real if not np.isnan(s_real) else s_unit
    line1 = (
        f"{meta['ticker']} – d={d}, s≈{s_disp:.6f}, λ≈{lam:.3g}, m̂≈{m_hat:.4f} "
        "(ajuste serie completa)"
    )
    line2 = (
        "RMSE_in(tr,val,ts,tv)="
        f"({rmse_in_train:.4f}, {rmse_in_val:.4f}, "
        f"{rmse_in_test:.4f}, {rmse_in_both:.4f})"
    )
    line3 = (
        "RMSEw_in(tr,val,ts,tv)="
        f"({rmsew_in_train:.4f}, {rmsew_in_val:.4f}, "
        f"{rmsew_in_test:.4f}, {rmsew_in_both:.4f})"
    )

    ax.set_title(line1 + "\n" + line2 + "\n" + line3)
    ax.set_xlabel("Días")
    ax.set_ylabel("Log-precio" if meta.get("use_log", True) else "Precio")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    # Optional save
    if save_pdf_path is not None:
        import os
        path_and_name = f"{save_pdf_path}/{meta['ticker']}/"
        os.makedirs(os.path.dirname(path_and_name), exist_ok=True)
        filename = f"{path_and_name}_FULLSERIES_NONPOLY_d_{d}_s_{s_disp:.3f}.pdf"
        plt.savefig(filename)

    plt.show()


# ============================================================
#  Example main: how to plug this after py12_timeweighted
# ============================================================

def main(
    ticker: str = "NVDA",
    start: str = "2010-01-01",
    save_pdf_path: Optional[str] = None,
    d_list: Sequence[int] = (1, 2, 3),
    best_s_by_d: Optional[Dict[int, float]] = None,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    min_train: int = 50,
    min_val: int = 50,
):
    """
    Example driver that shows how to call this module *after*
    py12_timeweighted.py.

    Typical workflow:
        1) Run py12_timeweighted.main(...) to search s on a grid and
           detect local minima for each d, with proper train/val/test.
        2) Extract a chosen s* per d (e.g., the s minimizing J_val).
        3) Call this main, passing best_s_by_d={d: s_star, ...} to see
           the corresponding non-polynomial full-series trend plots.

    Parameters
    ----------
    ticker, start, save_pdf_path : same semantics as in py12_timeweighted.
    d_list : iterable of int
        Differencing orders to visualize.
    best_s_by_d : dict or None
        If provided, maps each d in d_list to a chosen s*. If None,
        a dummy value s=0.99 is used for all d (for quick testing).
    frac_train, frac_val, min_train, min_val : float, float, int, int
        Parameters for the deterministic train/val/test split. Use the
        same values you used in py12_timeweighted to align segments.
    """
    # 1. Load series
    t_all, Z_all, meta = load_sp500_series(
        ticker=ticker,
        start=start,
        end=None,
        use_log=True,
        csv_path=None,
    )

    if save_pdf_path is not None:
        import os
        path_and_name = f"{save_pdf_path}/{meta['ticker']}/"
        os.makedirs(os.path.dirname(path_and_name), exist_ok=True)
        print(f"PDFs will be saved to: {path_and_name}")
    else:
        path_and_name = None

    print(
        f"Loaded {meta['ticker']} from {meta['start']} to {meta['end']} "
        f"(N={Z_all.size}, log={meta['use_log']})."
    )

    # 2. Split series deterministically, same as in py12_timeweighted
    Z_train, Z_val, Z_test, N_train, N_val, N_test = train_val_test_split(
        Z_all,
        frac_train=frac_train,
        frac_val=frac_val,
        min_train=min_train,
        min_val=min_val,
    )
    print(f"Split: N_train={N_train}, N_val={N_val}, N_test={N_test}")

    # 3. Choose s per d (ideally passed from py12_timeweighted)
    if best_s_by_d is None:
        # Dummy choice: very smooth curve. Replace with values from py12.
        best_s_by_d = {d: 0.99 for d in d_list}
        print(
            "Warning: best_s_by_d not provided; using s=0.99 for all d. "
            "For real analysis, pass best_s_by_d from py12_timeweighted."
        )

    # 4. Plot full-series smoothing for each d
    for d in d_list:
        if d not in best_s_by_d:
            print(f"Skipping d={d}: no s* in best_s_by_d.")
            continue
        s_star = float(best_s_by_d[d])
        print(
            f"\n--- Full-series smoothing for d={d}, s*≈{s_star:.6f} "
            "(non-polynomial tail) ---"
        )
        plot_fullseries_split_nonpoly(
            Z_all=Z_all,
            N_train=N_train,
            N_val=N_val,
            d=d,
            s_unit=s_star,
            meta=meta,
            save_pdf_path=path_and_name,
        )


if __name__ == "__main__":
    main()
