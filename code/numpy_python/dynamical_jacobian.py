#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamical_jacobian.py  --  Adaptive Largest Lyapunov Exponent estimator &
6-Gate Verification Pipeline for EMD Intrinsic Mode Functions (IMFs).

Implements:
  - Pre-Gate : Dominant period T_k estimation (PSD peak or zero-crossing),
               anti-aliased adaptive decimation D_k = max(1, floor(T_k / 20)),
               and minimum cycle / effective sample count checks.
  - Lag L    : Estimated from decimated correlation time via Average Mutual
               Information (ami), 1/e autocorrelation decay (exp_decay), or
               first autocorrelation zero-crossing (zero_cross).
  - Dim m    : Estimated per IMF via Cao's E1(m) method (cao) or False Nearest
               Neighbors (fnn).
  - Model Selection : Chronological 70/30 train/validation split (no shuffling)
               selecting (L*, m*, q*) that minimizes validation BIC:
                 BIC = N_val * ln(NMSE_val) + p * ln(N_val),  p = q*(m + 2) + 1.
  - Gate 1   : Out-of-sample predictive skill (NMSE_val <= threshold).
  - Gate 2   : Sampling-rate plausibility (lambda_orb < max_le_step in [0.5, 1.0]).
  - Gate 3   : Multi-start spread & hidden-unit capacity plateau (q* +/- 1, 2).
  - Gate 4   : Cross-dimension m invariance (Gencay & Dechert manifold check).
  - Gate 5   : Geometric cross-check via Rosenstein trajectory divergence on the
               reconstructed attractor (+ optional Parlitz time-reversal test).
  - Gate 6   : Nonlinearity confirmation via B = 19-39 IAAFT surrogates (p < 0.05).
  - Selection: Rescales survivors to physical units lambda_phys = lambda_orb / (D_k * dt)
               and picks the largest verified Lyapunov exponent.
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import argparse
import csv
import fnmatch
import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
import scipy.io
from scipy import signal
from scipy.spatial import cKDTree

# Worker-global state for ProcessPoolExecutor
_G = {}


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
def setup_logger(log_file=None, verbose=False):
    logger = logging.getLogger("dynamical_jacobian")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def available_cpus():
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm))
    return max(1, os.cpu_count() or 1)


def n_params(q, m):
    """Free parameters in a 1-hidden-layer MLP with m inputs and q hidden units."""
    return 1 + q * (m + 2)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_series(path, key=None, column=0):
    """Load a 1-D float64 time series from .mat (v5/v7 or v7.3 HDF5), .npy, or text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.asarray(np.load(path), dtype=np.float64)
    elif ext in (".csv", ".txt", ".dat"):
        data = np.loadtxt(
            path, dtype=np.float64, delimiter="," if ext == ".csv" else None
        )
    else:
        try:
            mat = scipy.io.loadmat(path)
            names = [k for k in mat if not k.startswith("__")]
            if key is None and len(names) == 1:
                key = names[0]
            if key not in mat:
                raise KeyError(
                    f"{path}: variable '{key}' not found; available: {names}"
                )
            data = np.asarray(mat[key], dtype=np.float64)
        except NotImplementedError:
            import h5py

            with h5py.File(path, "r") as f:
                names = list(f.keys())
                if key is None and len(names) == 1:
                    key = names[0]
                if key not in f:
                    raise KeyError(
                        f"{path}: variable '{key}' not found; available: {names}"
                    )
                data = np.asarray(f[key], dtype=np.float64).T

    data = np.squeeze(data)
    if data.ndim == 2:
        if data.shape[0] < data.shape[1]:
            data = data.T
        data = data[:, int(column)]
    if data.ndim != 1:
        raise ValueError(
            f"{path}: cannot reduce variable to 1-D array (shape {data.shape})"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: series contains NaN or Inf")
    return data.copy()


def collect_files(folder, pattern="*.mat", recursive=False):
    matches = []
    if recursive:
        for root, _, filenames in os.walk(folder):
            for fn in filenames:
                if fnmatch.fnmatch(fn, pattern):
                    matches.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            full = os.path.join(folder, fn)
            if os.path.isfile(full) and fnmatch.fnmatch(fn, pattern):
                matches.append(full)
    # Natural sort so IMF_2 precedes IMF_10
    import re

    def _nat_key(p):
        base = os.path.basename(p)
        return [
            int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", base)
        ]

    matches.sort(key=_nat_key)
    return matches


# ----------------------------------------------------------------------------
# Pre-Gate & Adaptive Embedding Estimators (T_k, D_k, L, m)
# ----------------------------------------------------------------------------
def estimate_dominant_period(x, method="psd"):
    """Estimate dominant oscillation period T_k in raw samples."""
    x0 = x - np.mean(x)
    N = len(x0)
    if method == "zcr":
        crossings = np.where((x0[:-1] < 0.0) & (x0[1:] >= 0.0))[0]
        if len(crossings) < 2:
            return float(N)
        return float((crossings[-1] - crossings[0]) / (len(crossings) - 1))

    # Welch PSD peak, ignoring DC and sub-2-cycle trend leakage
    nperseg = min(N, max(2048, N // 2))
    freqs, psd = signal.welch(x0, fs=1.0, nperseg=nperseg)
    valid = freqs >= (2.0 / N)
    if not np.any(valid):
        return float(N)
    idx = int(np.argmax(psd[valid]))
    f_peak = float(freqs[valid][idx])
    return (1.0 / f_peak) if f_peak > 0.0 else float(N)


def adaptive_decimate(x, D_k):
    """Downsample x by D_k using a zero-phase anti-aliasing low-pass filter."""
    if D_k <= 1:
        return x.copy()
    cutoff = min(0.8 / D_k, 0.95)
    sos = signal.butter(8, cutoff, btype="low", output="sos")
    # Ensure signal is long enough for sosfiltfilt padding
    if len(x) > 3 * 27:
        x_filt = signal.sosfiltfilt(sos, x)
    else:
        x_filt = x
    return x_filt[::D_k].copy()


def estimate_lag(x, method="ami", max_lag=None):
    """Estimate optimal delay lag L on the decimated signal."""
    N = len(x)
    x0 = x - np.mean(x)
    if max_lag is None:
        max_lag = max(5, min(N // 10, 150))
    max_lag = max(2, min(int(max_lag), N // 4))

    # Normalized autocorrelation via FFT
    nfft = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(x0, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[: max_lag + 1]
    acf /= max(float(acf[0]), 1e-30)

    if method == "exp_decay":
        below = np.where(acf[1:] <= math.exp(-1.0))[0]
        return int(below[0] + 1) if len(below) > 0 else max(1, max_lag // 2)

    if method == "zero_cross":
        below = np.where(acf[1:] <= 0.0)[0]
        return int(below[0] + 1) if len(below) > 0 else max(1, max_lag // 2)

    # Average Mutual Information (AMI)
    n_bins = max(8, min(32, int(round(2.0 * (N ** (1.0 / 3.0))))))
    lo, hi = float(np.min(x0)), float(np.max(x0))
    if hi <= lo:
        return 1
    edges = np.linspace(lo, hi + 1e-12, n_bins + 1)
    bins = np.clip(np.digitize(x0, edges) - 1, 0, n_bins - 1)

    mi = np.zeros(max_lag + 1, dtype=np.float64)
    for tau in range(1, max_lag + 1):
        b1 = bins[:-tau]
        b2 = bins[tau:]
        hist2d = np.bincount(b1 * n_bins + b2, minlength=n_bins * n_bins).reshape(
            n_bins, n_bins
        )
        pxy = hist2d / float(len(b1))
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        nz = pxy > 0.0
        mi[tau] = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))

    for tau in range(2, max_lag):
        if mi[tau] < mi[tau - 1] and mi[tau] <= mi[tau + 1]:
            return int(tau)

    # Fallback if AMI decreases monotonically
    below = np.where(acf[1:] <= math.exp(-1.0))[0]
    return int(below[0] + 1) if len(below) > 0 else 1


def _delay_embed(x, m, L):
    T = len(x) - (m - 1) * L
    return np.column_stack([x[(m - 1 - i) * L : (m - 1 - i) * L + T] for i in range(m)])


def estimate_dim_cao(x, L, m_min=2, m_max=6, theiler=None):
    """Estimate unfolding embedding dimension m via Cao's (1997) E1(m) statistic."""
    if theiler is None:
        theiler = max(L, 5)
    E = {}
    for m in range(1, m_max + 2):
        if len(x) - m * L <= theiler + 15:
            break
        Xm = _delay_embed(x, m, L)
        Xm1 = _delay_embed(x, m + 1, L)
        K = len(Xm1)
        Xm = Xm[:K]
        tree = cKDTree(Xm)
        k_query = min(K, 2 * theiler + 5)
        dists, idxs = tree.query(Xm, k=k_query, workers=-1)
        ratios = []
        for i in range(K):
            valid = np.where((np.abs(idxs[i] - i) > theiler) & (dists[i] > 1e-12))[0]
            if len(valid) > 0:
                j = idxs[i, valid[0]]
                d_m = float(np.max(np.abs(Xm[i] - Xm[j])))
                if d_m > 1e-12:
                    d_m1 = float(np.max(np.abs(Xm1[i] - Xm1[j])))
                    ratios.append(d_m1 / d_m)
        if ratios:
            E[m] = float(np.mean(ratios))

    E1 = {m: E[m + 1] / E[m] for m in E if (m + 1) in E and E[m] > 0.0}
    for m in range(m_min, m_max + 1):
        if m in E1 and E1[m] >= 0.90:
            return m, E1
    return min(m_max, max(m_min, 3)), E1


def estimate_dim_fnn(x, L, m_min=2, m_max=6, theiler=None, r_tol=15.0, a_tol=2.0):
    """Estimate unfolding embedding dimension m via False Nearest Neighbors (Kennel 1992)."""
    if theiler is None:
        theiler = max(L, 5)
    std_x = float(np.std(x))
    fnn_rates = {}
    for m in range(1, m_max + 1):
        if len(x) - m * L <= theiler + 15:
            break
        Xm = _delay_embed(x, m, L)
        Xm1 = _delay_embed(x, m + 1, L)
        K = len(Xm1)
        Xm = Xm[:K]
        tree = cKDTree(Xm)
        k_query = min(K, 2 * theiler + 5)
        dists, idxs = tree.query(Xm, k=k_query, workers=-1)
        false_cnt = 0
        total_cnt = 0
        for i in range(K):
            valid = np.where((np.abs(idxs[i] - i) > theiler) & (dists[i] > 1e-12))[0]
            if len(valid) > 0:
                j = idxs[i, valid[0]]
                r_m = float(dists[i, valid[0]])
                diff_new = abs(float(Xm1[i, 0] - Xm1[j, 0]))
                r_m1 = math.hypot(r_m, diff_new)
                is_false = (diff_new / r_m > r_tol) or (
                    r_m1 / max(std_x, 1e-12) > a_tol
                )
                false_cnt += int(is_false)
                total_cnt += 1
        if total_cnt > 0:
            fnn_rates[m] = false_cnt / float(total_cnt)

    for m in range(m_min, m_max + 1):
        if m in fnn_rates and fnn_rates[m] < 0.05:
            return m, fnn_rates
    return min(m_max, max(m_min, 3)), fnn_rates


# ----------------------------------------------------------------------------
# Neural network & Levenberg-Marquardt Jacobian estimator
# ----------------------------------------------------------------------------
def _build_problem(x, L, m, q):
    N = len(x)
    T = N - m * L
    if T <= 1:
        raise ValueError(f"Series too short (N={N}) for m={m}, L={L}")
    X = np.column_stack([x[(m - 1 - i) * L : (m - 1 - i) * L + T] for i in range(m)])
    y = x[m * L : m * L + T]
    return SimpleNamespace(L=L, m=m, q=q, X=X, y=y, T=T)


def _unpack(th, q, m):
    b0 = th[0]
    w = th[1 : 1 + q]
    c = th[1 + q : 1 + 2 * q]
    W = th[1 + 2 * q : 1 + q * (m + 2)].reshape(q, m)
    return b0, w, c, W


def _activate(Z, fn):
    if fn == "tanh":
        H = np.tanh(Z)
        dH = 1.0 - H * H
    elif fn in ("logistic", "sigmoid"):
        Zc = np.clip(Z, -60.0, 60.0)
        H = 1.0 / (1.0 + np.exp(-Zc))
        dH = H * (1.0 - H)
    else:
        raise ValueError(f"Unsupported activation '{fn}'")
    return H, dH


def _eval_res_and_jac(prob, th, fn):
    X, y, q, m = prob.X, prob.y, prob.q, prob.m
    T = prob.T
    b0, w, c, W = _unpack(th, q, m)

    Z = X @ W.T + c
    H, dH = _activate(Z, fn)
    r = (H @ w + b0) - y

    P = n_params(q, m)
    J = np.empty((T, P), dtype=np.float64)
    J[:, 0] = 1.0
    J[:, 1 : 1 + q] = H
    G = dH * w
    J[:, 1 + q : 1 + 2 * q] = G
    J[:, 1 + 2 * q :].reshape(T, q, m)[:] = G[:, :, None] * X[:, None, :]
    return r, J


def _eval_sse(prob, th, fn):
    b0, w, c, W = _unpack(th, prob.q, prob.m)
    Z = prob.X @ W.T + c
    if fn == "tanh":
        H = np.tanh(Z)
    else:
        H = 1.0 / (1.0 + np.exp(-np.clip(Z, -60.0, 60.0)))
    r = (H @ w + b0) - prob.y
    return float(np.dot(r, r))


def _rf_init(prob, rng, n_cand, fn, ridge=1e-8):
    X, y, q, m = prob.X, prob.y, prob.q, prob.m
    scales = (0.5, 1.0, 2.0)
    best_sse = math.inf
    best_th = None
    eye = ridge * np.eye(q + 1, dtype=np.float64)

    for k in range(max(1, n_cand)):
        sc = scales[k % len(scales)] / math.sqrt(m)
        W = rng.normal(0.0, sc, size=(q, m))
        c = rng.normal(0.0, 0.5, size=q)
        Z = X @ W.T + c
        H, _ = _activate(Z, fn)
        Phi = np.column_stack([np.ones(prob.T, dtype=np.float64), H])
        A = Phi.T @ Phi + eye
        rhs = Phi.T @ y
        try:
            beta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        r = Phi @ beta - y
        sse = float(np.dot(r, r))
        if sse < best_sse:
            best_sse = sse
            best_th = np.concatenate([[beta[0]], beta[1:], c, W.ravel()])
    return best_th


def run_lm(prob, th0, max_nfev, tol, fn="tanh"):
    th = th0.astype(np.float64, copy=True)
    P = len(th)
    eye = np.eye(P, dtype=np.float64)

    r, J = _eval_res_and_jac(prob, th, fn)
    sse = float(np.dot(r, r))
    nfev = 1
    mu = 1e-3

    while nfev < max_nfev:
        A = J.T @ J
        g = J.T @ r
        if np.max(np.abs(g)) <= tol:
            break

        accepted = False
        while nfev < max_nfev:
            try:
                step = -np.linalg.solve(A + mu * eye, g)
            except np.linalg.LinAlgError:
                mu *= 10.0
                if mu > 1e14:
                    break
                continue

            th_try = th + step
            sse_try = _eval_sse(prob, th_try, fn)
            nfev += 1

            if np.isfinite(sse_try) and sse_try < sse:
                rel_imp = (sse - sse_try) / max(sse, 1e-30)
                step_rel = np.linalg.norm(step) / max(np.linalg.norm(th), 1e-12)
                th = th_try
                sse = sse_try
                mu = max(mu * 0.33, 1e-15)
                accepted = True
                if rel_imp < tol and step_rel < math.sqrt(tol):
                    return th, sse
                r, J = _eval_res_and_jac(prob, th, fn)
                break
            else:
                mu *= 5.0
                if mu > 1e14:
                    break

        if not accepted or mu > 1e14:
            break

    return th, sse


def lyapunov_step(X, th, q, fn, stride=1):
    """Largest Lyapunov exponent per L-step map via QR decomposition."""
    X_sub = X[stride - 1 :: stride]
    K, m = X_sub.shape
    if K < 2:
        return float("nan")

    _, w, c, W = _unpack(th, q, m)
    Z = X_sub @ W.T + c
    _, dH = _activate(Z, fn)
    grads = (dH * w) @ W

    Q = np.eye(m, dtype=np.float64)
    lyap_sum = np.zeros(m, dtype=np.float64)

    for t in range(K):
        if m == 1:
            Z_mat = grads[t, 0] * Q
        else:
            Z_mat = np.vstack((grads[t] @ Q, Q[:-1, :]))

        Q, R = np.linalg.qr(Z_mat)
        diag_R = np.diag(R)
        signs = np.where(diag_R >= 0.0, 1.0, -1.0)
        Q = Q * signs
        lyap_sum += np.log(np.maximum(np.abs(diag_R), 1e-300))

    return float(lyap_sum[0] / K)


def lyapunov_orb(X, th, q, fn, L):
    """Orbit-consistent largest Lyapunov exponent PER DECIMATED SAMPLE."""
    if L == 1:
        lam_step = lyapunov_step(X, th, q, fn, stride=1)
    else:
        vals = [lyapunov_step(X[r:], th, q, fn, stride=L) for r in range(L)]
        if not all(np.isfinite(vals)):
            return float("nan")
        lam_step = float(np.mean(vals))
    return lam_step / float(L)


def fit_single_model(x_train, x_val, L, m, q, cfg, x_full=None):
    """
    Fit (L, m, q) on x_train (70%), score out-of-sample NMSE & BIC on x_val (30%),
    and optionally refit on x_full for full-trajectory lambda_orb & spread.
    """
    fn = cfg["activation"]
    prob_tr = _build_problem(x_train, L, m, q)
    prob_va = _build_problem(x_val, L, m, q)

    candidates = []
    for st in range(cfg["starts"]):
        ss = np.random.SeedSequence([cfg["seed"], L, m, q, st])
        rng = np.random.default_rng(ss)
        th0 = _rf_init(prob_tr, rng, cfg["rf_cand"], fn)
        th, sse_tr = run_lm(prob_tr, th0, cfg["screen_nfev"], cfg["screen_tol"], fn)
        if np.isfinite(sse_tr):
            candidates.append((sse_tr, th))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    polished = []
    for _, th0 in candidates[: cfg["topk"]]:
        th, sse_tr = run_lm(prob_tr, th0, cfg["polish_nfev"], cfg["polish_tol"], fn)
        if np.isfinite(sse_tr):
            polished.append((sse_tr, th))

    if not polished:
        return None

    polished.sort(key=lambda item: item[0])
    best_th_tr = polished[0][1]

    # Out-of-sample validation NMSE and BIC
    sse_val = _eval_sse(prob_va, best_th_tr, fn)
    var_val = max(float(np.var(prob_va.y)), 1e-30)
    nmse_val = sse_val / (prob_va.T * var_val)
    p = n_params(q, m)
    n_val = prob_va.T
    bic_val = n_val * math.log(max(nmse_val, 1e-30)) + p * math.log(max(n_val, 2))

    # Compute lambda_orb and multi-start spread (on x_full if provided, else x_train)
    if x_full is not None:
        prob_full = _build_problem(x_full, L, m, q)
        full_lams = []
        for _, th_init in polished:
            th_f, sse_f = run_lm(
                prob_full, th_init, cfg["polish_nfev"], cfg["polish_tol"], fn
            )
            if np.isfinite(sse_f):
                lam_f = lyapunov_orb(prob_full.X, th_f, q, fn, L)
                if np.isfinite(lam_f):
                    full_lams.append((sse_f, lam_f))
        if full_lams:
            full_lams.sort(key=lambda item: item[0])
            lam_best = full_lams[0][1]
            vals = [v[1] for v in full_lams]
            spread = max(vals) - min(vals)
        else:
            lam_best = lyapunov_orb(prob_tr.X, best_th_tr, q, fn, L)
            spread = float("nan")
    else:
        tr_lams = [lyapunov_orb(prob_tr.X, th_p, q, fn, L) for _, th_p in polished]
        tr_lams = [v for v in tr_lams if np.isfinite(v)]
        lam_best = tr_lams[0] if tr_lams else float("nan")
        spread = (max(tr_lams) - min(tr_lams)) if tr_lams else float("nan")

    return {
        "L": int(L),
        "m": int(m),
        "q": int(q),
        "n_par": int(p),
        "n_val": int(n_val),
        "nmse_val": float(nmse_val),
        "bic_val": float(bic_val),
        "lambda_orb": float(lam_best),
        "spread": float(spread),
    }


# ----------------------------------------------------------------------------
# Parallel tasks for BIC grid & IAAFT Surrogates
# ----------------------------------------------------------------------------
def _init_grid_worker(x_train, x_val, cfg):
    _G["x_train"] = x_train
    _G["x_val"] = x_val
    _G["cfg"] = cfg


def _grid_combo_task(combo):
    L, m, q = combo
    return fit_single_model(_G["x_train"], _G["x_val"], L, m, q, _G["cfg"], x_full=None)


def _init_surr_worker(surrogates, L, m, q, train_ratio, cfg):
    _G["surrogates"] = surrogates
    _G["L"] = L
    _G["m"] = m
    _G["q"] = q
    _G["train_ratio"] = train_ratio
    _G["cfg"] = cfg


def _surrogate_task(idx):
    s = _G["surrogates"][idx]
    split = int(len(s) * _G["train_ratio"])
    s_tr = s[:split]
    s_va = s[split:]
    mu_tr = float(np.mean(s_tr))
    sd_tr = max(float(np.std(s_tr)), 1e-12)
    s_tr_n = (s_tr - mu_tr) / sd_tr
    s_va_n = (s_va - mu_tr) / sd_tr
    s_full_n = (s - float(np.mean(s))) / max(float(np.std(s)), 1e-12)

    cfg = dict(_G["cfg"])
    cfg["seed"] = cfg["seed"] + 10000 + idx
    res = fit_single_model(
        s_tr_n, s_va_n, _G["L"], _G["m"], _G["q"], cfg, x_full=s_full_n
    )
    if res is None:
        return idx, float("nan"), float("nan")
    return idx, res["nmse_val"], res["lambda_orb"]


# ----------------------------------------------------------------------------
# Gate 5 & Gate 6 Helpers (Rosenstein Trajectory Divergence & IAAFT Surrogates)
# ----------------------------------------------------------------------------
def rosenstein_trajectory_le(x, L, m, theiler=None):
    """
    Non-Jacobian largest Lyapunov exponent (per decimated sample) computed from
    the divergence rate of nearest-neighbor trajectories on the reconstructed attractor.
    """
    if theiler is None:
        theiler = max(L * m, 10)
    X = _delay_embed(x, m, L)
    K = len(X)
    max_step = min(40, K // 10)
    if K <= theiler + max_step + 5 or max_step < 5:
        return float("nan")

    tree = cKDTree(X[:-max_step])
    k_query = min(K - max_step, 2 * theiler + 5)
    dists, idxs = tree.query(X[:-max_step], k=k_query, workers=-1)

    nn_idx = np.full(K - max_step, -1, dtype=int)
    for i in range(K - max_step):
        valid = np.where((np.abs(idxs[i] - i) > theiler) & (dists[i] > 1e-12))[0]
        if len(valid) > 0:
            nn_idx[i] = idxs[i, valid[0]]

    valid_i = np.where(nn_idx >= 0)[0]
    if len(valid_i) < 20:
        return float("nan")

    div = np.zeros(max_step + 1, dtype=np.float64)
    for s in range(max_step + 1):
        d = np.linalg.norm(X[valid_i + s] - X[nn_idx[valid_i] + s], axis=1)
        div[s] = float(np.mean(np.log(np.maximum(d, 1e-15))))

    sat_level = div[0] + 0.85 * (float(np.max(div)) - div[0])
    end_s = min(15, max_step)
    for s in range(4, max_step + 1):
        if div[s] >= sat_level:
            end_s = s
            break

    steps = np.arange(1, end_s + 1, dtype=np.float64)
    slope = float(np.polyfit(steps, div[1 : end_s + 1], 1)[0])
    return slope


def generate_iaaft_surrogates(x, n_surr=19, seed=0, max_iter=60):
    """
    Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogates:
    preserves exact power spectrum and exact amplitude distribution while
    destroying nonlinear phase coupling.
    """
    rng = np.random.default_rng(seed)
    N = len(x)
    x_sorted = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    surrogates = []

    for _ in range(n_surr):
        s = rng.permutation(x)
        prev_ranks = None
        for _ in range(max_iter):
            s_fft = np.fft.rfft(s)
            s_matched = np.fft.irfft(amp * np.exp(1j * np.angle(s_fft)), n=N)
            ranks = np.argsort(np.argsort(s_matched))
            s = x_sorted[ranks]
            if prev_ranks is not None and np.array_equal(ranks, prev_ranks):
                break
            prev_ranks = ranks
        surrogates.append(s)
    return surrogates


# ----------------------------------------------------------------------------
# Full Single-IMF Pipeline (Pre-Gate + Adaptive Estimation + Gates 1..6)
# ----------------------------------------------------------------------------
def analyze_single_imf(filepath, args, logger):
    fname = os.path.basename(filepath)
    t_start = time.time()
    logger.info("=" * 78)
    logger.info("Processing: %s", fname)

    # ------------------------------------------------------------------
    # STEP 1: Load signal & Pre-Gate (Dominant Period + Adaptive Decimation)
    # ------------------------------------------------------------------
    x_raw = load_series(filepath, key=args.key, column=args.column)
    n_raw = len(x_raw)
    if float(np.std(x_raw)) == 0.0:
        logger.warning("  [PRE-GATE] REJECTED: Signal is constant (std=0).")
        return _rejected_record(filepath, n_raw, "PRE-GATE", "Constant signal (std=0)")

    T_k = estimate_dominant_period(x_raw, method=args.period_method)
    D_k = max(1, int(math.floor(T_k / float(args.target_spc))))
    x_dec = adaptive_decimate(x_raw, D_k)
    n_eff = len(x_dec)
    n_cycles = n_raw / max(T_k, 1.0)
    tau_eff = float(args.dt) * D_k

    logger.info(
        "  [STEP 1: PRE-GATE] T_k (%s) = %.1f samples | D_k = %d -> N_eff = %d | "
        "cycles = %.1f | tau_eff = %g",
        args.period_method,
        T_k,
        D_k,
        n_eff,
        n_cycles,
        tau_eff,
    )

    if n_eff < args.min_samples or n_cycles < args.min_cycles:
        reason = (
            f"Insufficient recurrence: N_eff={n_eff} (<{args.min_samples}) or "
            f"cycles={n_cycles:.1f} (<{args.min_cycles})"
        )
        logger.warning("  [PRE-GATE] REJECTED: %s", reason)
        return _rejected_record(
            filepath,
            n_raw,
            "PRE-GATE",
            reason,
            T_k=T_k,
            D_k=D_k,
            n_eff=n_eff,
            n_cycles=n_cycles,
            tau_eff=tau_eff,
        )
    logger.info("  [PRE-GATE] PASSED.")

    # ------------------------------------------------------------------
    # STEP 2: Lag L & Unfolding Dimension m on Decimated IMF
    # ------------------------------------------------------------------
    T_k_dec = T_k / float(D_k)
    max_lag = max(4, min(n_eff // 10, int(round(T_k_dec))))
    if D_k > 1:
        L_est = 1
    else:
        L_est = estimate_lag(x_dec, method=args.lag_method, max_lag=max_lag)
        L_est = max(1, min(L_est, max(1, int(round(T_k_dec / 2.0)))))

    theiler = max(L_est, int(round(T_k_dec / 2.0)), 5)
    if args.dim_method == "fnn":
        m_est, dim_diag = estimate_dim_fnn(
            x_dec, L_est, m_min=args.m_min, m_max=args.m_max, theiler=theiler
        )
    else:
        m_est, dim_diag = estimate_dim_cao(
            x_dec, L_est, m_min=args.m_min, m_max=args.m_max, theiler=theiler
        )

    logger.info(
        "  [STEP 2: EMBEDDING] L_est (%s) = %d | m_est (%s) = %d (diag=%s)",
        args.lag_method,
        L_est,
        args.dim_method,
        m_est,
        {k: round(v, 3) for k, v in dim_diag.items()},
    )

    # ------------------------------------------------------------------
    # STEP 3: Chronological 70/30 Split & BIC Model Selection
    # ------------------------------------------------------------------
    split_idx = int(n_eff * args.train_ratio)
    x_tr_raw = x_dec[:split_idx]
    x_va_raw = x_dec[split_idx:]
    mu_tr = float(np.mean(x_tr_raw))
    sd_tr = max(float(np.std(x_tr_raw)), 1e-12)
    x_train = (x_tr_raw - mu_tr) / sd_tr
    x_val = (x_va_raw - mu_tr) / sd_tr
    x_full = (x_dec - float(np.mean(x_dec))) / max(float(np.std(x_dec)), 1e-12)

    lag_candidates = sorted(
        {max(1, L_est - 1), L_est, L_est + 1} if args.search_lag_neighbors else {L_est}
    )
    m_candidates = list(range(args.m_min, args.m_max + 1))
    q_candidates = args.qs

    combos = [
        (L, m, q)
        for L in lag_candidates
        for m in m_candidates
        for q in q_candidates
        if (len(x_train) - m * L > n_params(q, m) + 5) and (len(x_val) - m * L > 10)
    ]
    combos.sort(key=lambda c: -n_params(c[2], c[1]))

    if not combos:
        reason = "No valid (L, m, q) combinations fit chronological train/val split"
        logger.warning("  [STEP 3: BIC] REJECTED: %s", reason)
        return _rejected_record(
            filepath, n_raw, "BIC-GRID", reason, T_k=T_k, D_k=D_k, n_eff=n_eff
        )

    logger.info(
        "  [STEP 3: BIC] Evaluating %d (L,m,q) combos on chronological %.0f/%.0f split "
        "(N_tr=%d, N_val=%d)...",
        len(combos),
        100 * args.train_ratio,
        100 * (1.0 - args.train_ratio),
        len(x_train),
        len(x_val),
    )

    fit_cfg = dict(
        activation=args.activation,
        seed=args.seed,
        starts=args.starts,
        topk=args.topk,
        rf_cand=args.rf_cand,
        screen_nfev=args.screen_nfev,
        screen_tol=args.screen_tol,
        polish_nfev=args.polish_nfev,
        polish_tol=args.polish_tol,
    )

    grid_results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_grid_worker,
        initargs=(x_train, x_val, fit_cfg),
    ) as pool:
        for res in pool.map(_grid_combo_task, combos):
            if res is not None:
                grid_results[(res["L"], res["m"], res["q"])] = res

    if not grid_results:
        reason = "All LM optimization runs failed to converge"
        logger.warning("  [STEP 3: BIC] REJECTED: %s", reason)
        return _rejected_record(
            filepath, n_raw, "BIC-GRID", reason, T_k=T_k, D_k=D_k, n_eff=n_eff
        )

    best_bic_row = min(grid_results.values(), key=lambda r: r["bic_val"])
    L_star, m_star, q_star = best_bic_row["L"], best_bic_row["m"], best_bic_row["q"]

    # Refit winning (L*, m*, q*) on full decimated series for final lambda_orb & spread
    best_full = fit_single_model(
        x_train, x_val, L_star, m_star, q_star, fit_cfg, x_full=x_full
    )
    lam_orb = best_full["lambda_orb"]
    spread = best_full["spread"]
    nmse_val = best_full["nmse_val"]
    bic_val = best_full["bic_val"]
    lam_phys = lam_orb / tau_eff
    spread_phys = spread / tau_eff

    logger.info(
        "  [STEP 3: BIC] Selected (L*=%d, m*=%d, q*=%d) | BIC=%.2f | NMSE_val=%.3e | "
        "lambda_orb=%+.5f/step | lambda_phys=%+.5f/time | spread=%.4f",
        L_star,
        m_star,
        q_star,
        bic_val,
        nmse_val,
        lam_orb,
        lam_phys,
        spread,
    )

    # ------------------------------------------------------------------
    # GATE 1: Out-of-Sample Predictive Skill (NMSE_val)
    # ------------------------------------------------------------------
    g1_pass = bool(np.isfinite(nmse_val) and nmse_val <= args.max_nmse)
    logger.info(
        "  [GATE 1: NMSE]     NMSE_val = %.3e (threshold <= %.3e) -> %s",
        nmse_val,
        args.max_nmse,
        "PASS" if g1_pass else "FAIL",
    )

    # ------------------------------------------------------------------
    # GATE 2: Sampling-Rate Plausibility (lambda_orb < max_le_step)
    # ------------------------------------------------------------------
    g2_pass = bool(np.isfinite(lam_orb) and abs(lam_orb) < args.max_le_step)
    logger.info(
        "  [GATE 2: RATE]     |lambda_orb| = %.5f (threshold < %.2f /step) -> %s",
        abs(lam_orb),
        args.max_le_step,
        "PASS" if g2_pass else "FAIL",
    )

    # ------------------------------------------------------------------
    # GATE 3: Multi-Start Spread & Hidden-Unit Capacity Plateau (q* +/- 1, 2)
    # ------------------------------------------------------------------
    spread_limit = max(0.25 * abs(lam_orb), args.min_spread_floor)
    spread_ok = bool(np.isfinite(spread) and spread <= spread_limit)

    q_neighbors = [
        q
        for q in (q_star - 2, q_star - 1, q_star + 1, q_star + 2)
        if q >= 2 and (L_star, m_star, q) in grid_results
    ]
    q_lams = [grid_results[(L_star, m_star, q)]["lambda_orb"] for q in q_neighbors]
    q_base = best_bic_row["lambda_orb"]
    if q_lams:
        max_q_jump = max(abs(ql - q_base) for ql in q_lams)
        q_tol = max(0.35 * abs(q_base), 0.02)
        plateau_ok = bool(max_q_jump <= q_tol)
    else:
        max_q_jump = 0.0
        plateau_ok = True

    g3_pass = spread_ok and plateau_ok
    logger.info(
        "  [GATE 3: PLATEAU]  spread=%.4f (<=%.4f: %s) | q-neighbors jump=%.4f (%s) -> %s",
        spread,
        spread_limit,
        "OK" if spread_ok else "HIGH",
        max_q_jump,
        "OK" if plateau_ok else "ERRATIC",
        "PASS" if g3_pass else "FAIL",
    )

    # ------------------------------------------------------------------
    # GATE 4: Cross-Dimension (m) Invariance (Gencay & Dechert Manifold Check)
    # ------------------------------------------------------------------
    m_slice = sorted(
        [
            grid_results[(L_star, m, q_star)]
            for m in m_candidates
            if (L_star, m, q_star) in grid_results
        ],
        key=lambda r: r["m"],
    )
    # Check dims >= max(m_est, m_star) for spurious upward jumps without NMSE improvement
    g4_pass = True
    m_jump_detail = "stable"
    ref_rows = [r for r in m_slice if r["m"] >= min(m_est, m_star)]
    if len(ref_rows) >= 2:
        base_r = ref_rows[0]
        for r_hi in ref_rows[1:]:
            nmse_improved = r_hi["nmse_val"] < 0.80 * base_r["nmse_val"]
            upward_jump = r_hi["lambda_orb"] - base_r["lambda_orb"]
            jump_thresh = max(0.30 * abs(base_r["lambda_orb"]), 0.02)
            if (not nmse_improved) and (upward_jump > jump_thresh):
                g4_pass = False
                m_jump_detail = (
                    f"spurious jump +{(upward_jump):.4f} at m={r_hi['m']} "
                    f"(NMSE {base_r['nmse_val']:.2e}->{r_hi['nmse_val']:.2e})"
                )
                break

    logger.info(
        "  [GATE 4: MANIFOLD] Gencay-Dechert dim sweep (%s) -> %s",
        m_jump_detail,
        "PASS" if g4_pass else "FAIL",
    )

    # ------------------------------------------------------------------
    # GATE 5: Geometric Cross-Check (Rosenstein Trajectory Divergence + Parlitz)
    # ------------------------------------------------------------------
    lam_traj = rosenstein_trajectory_le(x_full, L_star, m_star, theiler=theiler)
    if np.isfinite(lam_traj):
        # Reject if Jacobian lambda substantially exceeds manifold trajectory divergence
        allowed_upper = max(1.5 * lam_traj + 0.02, lam_traj + 0.05)
        g5_geom_ok = bool(lam_orb <= allowed_upper)
    else:
        g5_geom_ok = True

    parlitz_msg = ""
    g5_parlitz_ok = True
    if args.parlitz_check:
        x_rev = x_full[::-1].copy()
        sp_rev = int(len(x_rev) * args.train_ratio)
        rev_res = fit_single_model(
            x_rev[:sp_rev],
            x_rev[sp_rev:],
            L_star,
            m_star,
            q_star,
            fit_cfg,
            x_full=x_rev,
        )
        lam_rev = rev_res["lambda_orb"] if rev_res else float("nan")
        if np.isfinite(lam_rev) and abs(lam_orb) > 0.01:
            # Time-reversed LE should flip sign: lam_orb + lam_rev ~ 0
            g5_parlitz_ok = bool(
                abs(lam_orb + lam_rev) <= max(0.5 * abs(lam_orb), 0.03)
            )
        parlitz_msg = f" | Parlitz lam_rev={lam_rev:+.5f} ({'OK' if g5_parlitz_ok else 'MISMATCH'})"

    g5_pass = g5_geom_ok and g5_parlitz_ok
    logger.info(
        "  [GATE 5: GEOMETRY] Jacobian=%.5f vs Trajectory(Rosenstein)=%.5f%s -> %s",
        lam_orb,
        lam_traj,
        parlitz_msg,
        "PASS" if g5_pass else "FAIL",
    )

    # ------------------------------------------------------------------
    # GATE 6: Nonlinearity Confirmation via IAAFT Surrogates
    # ------------------------------------------------------------------
    # Skip expensive surrogate loop if earlier gates already failed (unless --force-surrogates)
    prior_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
    p_nmse = float("nan")
    p_lam = float("nan")
    surr_nmse_mean = float("nan")
    surr_lam_mean = float("nan")

    if not prior_pass and not args.force_surrogates:
        g6_pass = False
        logger.info(
            "  [GATE 6: IAAFT]    Skipped (already failed an earlier gate) -> FAIL"
        )
    else:
        logger.info(
            "  [GATE 6: IAAFT]    Generating & fitting B=%d IAAFT surrogates...",
            args.n_surrogates,
        )
        surrogates = generate_iaaft_surrogates(
            x_dec, n_surr=args.n_surrogates, seed=args.seed + 777
        )
        surr_nmses = []
        surr_lams = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_surr_worker,
            initargs=(surrogates, L_star, m_star, q_star, args.train_ratio, fit_cfg),
        ) as pool:
            for _, s_nmse, s_lam in pool.map(_surrogate_task, range(args.n_surrogates)):
                if np.isfinite(s_nmse):
                    surr_nmses.append(s_nmse)
                if np.isfinite(s_lam):
                    surr_lams.append(s_lam)

        if len(surr_nmses) >= 5 and len(surr_lams) >= 5:
            surr_nmse_arr = np.asarray(surr_nmses)
            surr_lam_arr = np.asarray(surr_lams)
            surr_nmse_mean = float(np.mean(surr_nmse_arr))
            surr_lam_mean = float(np.mean(surr_lam_arr))

            # One-sided empirical p-values: real NMSE must be lower, real lambda higher
            p_nmse = (1.0 + float(np.sum(surr_nmse_arr <= nmse_val))) / (
                len(surr_nmse_arr) + 1.0
            )
            p_lam = (1.0 + float(np.sum(surr_lam_arr >= lam_orb))) / (
                len(surr_lam_arr) + 1.0
            )
            g6_pass = bool((p_nmse <= args.surr_alpha) and (p_lam <= args.surr_alpha))
        else:
            g6_pass = False

        logger.info(
            "  [GATE 6: IAAFT]    NMSE_val=%.3e vs surr=%.3e (p=%.3f) | "
            "LE=%.5f vs surr=%.5f (p=%.3f) -> %s",
            nmse_val,
            surr_nmse_mean,
            p_nmse,
            lam_orb,
            surr_lam_mean,
            p_lam,
            "PASS" if g6_pass else "FAIL",
        )

    all_passed = prior_pass and g6_pass
    failed_gates = [
        name
        for name, ok in [
            ("G1_NMSE", g1_pass),
            ("G2_RATE", g2_pass),
            ("G3_PLATEAU", g3_pass),
            ("G4_MANIFOLD", g4_pass),
            ("G5_GEOMETRY", g5_pass),
            ("G6_IAAFT", g6_pass),
        ]
        if not ok
    ]

    logger.info(
        "  [VERDICT] %s -> %s (lambda_phys = %+.6f +/- %.6f /time) [%.1fs]",
        fname,
        "VERIFIED CHAOTIC" if all_passed else f"REJECTED ({','.join(failed_gates)})",
        lam_phys,
        spread_phys if np.isfinite(spread_phys) else 0.0,
        time.time() - t_start,
    )

    return {
        "file": fname,
        "input": filepath,
        "verified": bool(all_passed),
        "failed_gates": ",".join(failed_gates),
        "N_raw": int(n_raw),
        "T_k": float(T_k),
        "D_k": int(D_k),
        "N_eff": int(n_eff),
        "n_cycles": float(n_cycles),
        "dt": float(args.dt),
        "tau_eff": float(tau_eff),
        "L_star": int(L_star),
        "m_est": int(m_est),
        "m_star": int(m_star),
        "q_star": int(q_star),
        "bic_val": float(bic_val),
        "nmse_val": float(nmse_val),
        "lambda_orb": float(lam_orb),
        "spread_orb": float(spread),
        "lambda_traj": float(lam_traj),
        "lambda_phys": float(lam_phys),
        "spread_phys": float(spread_phys),
        "p_nmse_iaaft": float(p_nmse),
        "p_lam_iaaft": float(p_lam),
        "grid_rows": list(grid_results.values()),
    }


def _rejected_record(
    filepath,
    n_raw,
    stage,
    reason,
    T_k=float("nan"),
    D_k=1,
    n_eff=0,
    n_cycles=float("nan"),
    tau_eff=float("nan"),
):
    return {
        "file": os.path.basename(filepath),
        "input": filepath,
        "verified": False,
        "failed_gates": f"{stage}:{reason}",
        "N_raw": int(n_raw),
        "T_k": float(T_k),
        "D_k": int(D_k),
        "N_eff": int(n_eff),
        "n_cycles": float(n_cycles),
        "dt": float("nan"),
        "tau_eff": float(tau_eff),
        "L_star": None,
        "m_est": None,
        "m_star": None,
        "q_star": None,
        "bic_val": float("nan"),
        "nmse_val": float("nan"),
        "lambda_orb": float("nan"),
        "spread_orb": float("nan"),
        "lambda_traj": float("nan"),
        "lambda_phys": float("nan"),
        "spread_phys": float("nan"),
        "p_nmse_iaaft": float("nan"),
        "p_lam_iaaft": float("nan"),
        "grid_rows": [],
    }


# ----------------------------------------------------------------------------
# CLI & Batch Driver
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to an IMF file (.mat/.npy/.csv) or a folder containing IMF .mat files",
    )
    p.add_argument(
        "--pattern",
        default="*.mat",
        help="Filename glob pattern when --input is a folder (default '*.mat')",
    )
    p.add_argument("--recursive", action="store_true", help="Search subfolders")
    p.add_argument("--key", default=None, help="Variable name inside .mat files")
    p.add_argument("--column", type=int, default=0, help="Column index if 2-D array")
    p.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Original sampling interval Delta t of raw IMFs (default 1.0)",
    )

    # Pre-Gate & Adaptive Embedding options
    pg = p.add_argument_group("Pre-Gate & adaptive embedding")
    pg.add_argument(
        "--period-method",
        choices=["psd", "zcr"],
        default="psd",
        help="Method to estimate dominant period T_k: Welch PSD peak ('psd') or zero-crossing ('zcr')",
    )
    pg.add_argument(
        "--target-spc",
        type=int,
        default=20,
        help="Target samples per cycle after adaptive decimation D_k = max(1, floor(T_k / target_spc)) (default 20)",
    )
    pg.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum effective decimated length N_eff required (default 500)",
    )
    pg.add_argument(
        "--min-cycles",
        type=float,
        default=20.0,
        help="Minimum full oscillation cycles required in record (default 20.0)",
    )
    pg.add_argument(
        "--lag-method",
        choices=["ami", "exp_decay", "zero_cross"],
        default="ami",
        help="Correlation-time method for lag L: 'ami', 'exp_decay' (1/e), or 'zero_cross' (default 'ami')",
    )
    pg.add_argument(
        "--search-lag-neighbors",
        action="store_true",
        help="Also include L_est +/- 1 in the BIC candidate grid",
    )
    pg.add_argument(
        "--dim-method",
        choices=["cao", "fnn"],
        default="cao",
        help="Method to estimate unfolding dimension m: 'cao' or 'fnn' (default 'cao')",
    )
    pg.add_argument(
        "--m-min", type=int, default=2, help="Minimum embedding dim m (default 2)"
    )
    pg.add_argument(
        "--m-max", type=int, default=6, help="Maximum embedding dim m (default 6)"
    )
    pg.add_argument(
        "--qs",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10],
        help="Candidate hidden-unit counts q (default: 2 3 4 5 6 8 10; avoids q=1 underfitting)",
    )
    pg.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological train split fraction for BIC model selection (default 0.70)",
    )

    # Gates 1..6 thresholds
    gt = p.add_argument_group("Verification gates (Gates 1-6)")
    gt.add_argument(
        "--max-nmse",
        type=float,
        default=0.10,
        help="Gate 1: Maximum allowed out-of-sample NMSE_val (default 0.10)",
    )
    gt.add_argument(
        "--max-le-step",
        type=float,
        default=0.50,
        help="Gate 2: Maximum plausible |lambda_orb| per decimated sample in [0.5, 1.0] (default 0.50)",
    )
    gt.add_argument(
        "--min-spread-floor",
        type=float,
        default=0.01,
        help="Gate 3: Minimum absolute spread tolerance when |lambda_orb| is near zero (default 0.01)",
    )
    gt.add_argument(
        "--parlitz-check",
        action="store_true",
        help="Gate 5: Also run Parlitz time-reversal sign-flip check lambda(-t) ~ -lambda(t)",
    )
    gt.add_argument(
        "--n-surrogates",
        type=int,
        default=19,
        help="Gate 6: Number of IAAFT surrogates B (recommended 19-39, default 19)",
    )
    gt.add_argument(
        "--surr-alpha",
        type=float,
        default=0.05,
        help="Gate 6: Significance level p-value threshold for IAAFT test (default 0.05)",
    )
    gt.add_argument(
        "--force-surrogates",
        action="store_true",
        help="Run Gate 6 IAAFT surrogates even on IMFs that failed Gates 1-5",
    )

    # Optimization & I/O
    op = p.add_argument_group("Optimization & outputs")
    op.add_argument(
        "--activation", choices=["tanh", "logistic", "sigmoid"], default="tanh"
    )
    op.add_argument(
        "--starts", type=int, default=10, help="Random starts per (L,m,q) (default 10)"
    )
    op.add_argument(
        "--rf-cand", type=int, default=30, help="Random-feature candidates (default 30)"
    )
    op.add_argument("--screen-nfev", type=int, default=250)
    op.add_argument("--screen-tol", type=float, default=1e-9)
    op.add_argument("--topk", type=int, default=2)
    op.add_argument("--polish-nfev", type=int, default=1500)
    op.add_argument("--polish-tol", type=float, default=1e-11)
    op.add_argument("--seed", type=int, default=0)
    op.add_argument("--workers", type=int, default=available_cpus())
    op.add_argument(
        "--out",
        default="dynamical_imf_results",
        help="Output folder for summary CSV/JSON and per-IMF details",
    )
    op.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    logger = setup_logger(os.path.join(args.out, "run.log"), verbose=args.verbose)

    if os.path.isdir(args.input):
        files = collect_files(
            args.input, pattern=args.pattern, recursive=args.recursive
        )
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        raise SystemExit(f"Input path not found: {args.input}")

    if not files:
        raise SystemExit(f"No matching files found in '{args.input}'.")

    logger.info(
        "Starting adaptive IMF Lyapunov pipeline on %d file(s) | workers=%d | "
        "period=%s | lag=%s | dim=%s | max_le_step=%.2f | surrogates=%d",
        len(files),
        args.workers,
        args.period_method,
        args.lag_method,
        args.dim_method,
        args.max_le_step,
        args.n_surrogates,
    )

    t_all = time.time()
    records = []
    for fp in files:
        try:
            rec = analyze_single_imf(fp, args, logger)
        except Exception as exc:
            logger.exception("Unhandled error processing %s: %s", fp, exc)
            rec = _rejected_record(fp, 0, "ERROR", str(exc))
        records.append(rec)

        stem = os.path.splitext(os.path.basename(fp))[0]
        with open(os.path.join(args.out, f"{stem}_details.json"), "w") as f:
            json.dump(rec, f, indent=2, default=_json_default)

    # Save consolidated CSV & JSON summaries
    summary_fields = [
        "file",
        "verified",
        "failed_gates",
        "N_raw",
        "T_k",
        "D_k",
        "N_eff",
        "n_cycles",
        "tau_eff",
        "L_star",
        "m_est",
        "m_star",
        "q_star",
        "bic_val",
        "nmse_val",
        "lambda_orb",
        "spread_orb",
        "lambda_traj",
        "lambda_phys",
        "spread_phys",
        "p_nmse_iaaft",
        "p_lam_iaaft",
    ]
    csv_path = os.path.join(args.out, "imf_verification_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Final selection among verified IMFs
    verified = [r for r in records if r["verified"] and np.isfinite(r["lambda_phys"])]

    logger.info("=" * 78)
    logger.info("FINAL IMF VERIFICATION SUMMARY (elapsed %.1fs)", time.time() - t_all)
    logger.info("=" * 78)
    for r in records:
        status = "PASS" if r["verified"] else f"FAIL [{r['failed_gates']}]"
        le_str = (
            f"{r['lambda_phys']:+.6f} +/- {r['spread_phys']:.6f}"
            if np.isfinite(r["lambda_phys"])
            else "N/A"
        )
        logger.info("%-18s | LE_phys = %-24s | %s", r["file"], le_str, status)

    winner = None
    if verified:
        winner = max(verified, key=lambda r: r["lambda_phys"])
        logger.info("-" * 78)
        logger.info(
            "SELECTED MAX TRUSTWORTHY LE: %s -> lambda_phys = %+.6f +/- %.6f /time "
            "(D_k=%d, L*=%d, m*=%d, q*=%d, NMSE_val=%.2e, p_IAAFT=%.3f)",
            winner["file"],
            winner["lambda_phys"],
            winner["spread_phys"],
            winner["D_k"],
            winner["L_star"],
            winner["m_star"],
            winner["q_star"],
            winner["nmse_val"],
            winner["p_lam_iaaft"],
        )
    else:
        logger.info("-" * 78)
        logger.info(
            "CONCLUSION: No IMF passed all 6 verification gates. "
            "Signal is consistent with stochastic/linear noise (no trustworthy deterministic chaos)."
        )

    with open(os.path.join(args.out, "imf_verification_summary.json"), "w") as f:
        json.dump(
            {
                "selected_winner": winner,
                "verified_count": len(verified),
                "total_files": len(records),
                "summary": [{k: r.get(k) for k in summary_fields} for r in records],
            },
            f,
            indent=2,
            default=_json_default,
        )
    logger.info("Saved summary CSV/JSON and run.log to '%s/'", args.out)


if __name__ == "__main__":
    main()
