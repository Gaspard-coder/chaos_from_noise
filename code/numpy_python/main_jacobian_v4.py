#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_jacobian_v4.py  --  Largest Lyapunov Exponent estimator via single-hidden-layer
neural network + analytic companion Jacobians (BenSaida method, v4).

Features:
  1. Single-file OR batch-folder processing via --input (processes every .mat file
     when --input points to a directory).
  2. Orbit-consistent (`orb`) Lyapunov estimator: chains Jacobians along the true
     L-step orbit (rows r, r+L, r+2L, ...) and averages over all L interleaved
     sub-orbits.
  3. Proper normalization: divides the per-map-step exponent by L * (dt * decimate)
     so all reported values are per TIME UNIT (or per sample when dt=1, decimate=1).
  4. Optional --auto-dt flag to automatically assign benchmark dt & decimate values
     from filenames (e.g., lorenz_x -> dt=0.02, decimate=5; rossler_x -> dt=0.1,
     decimate=5; discrete maps/noise -> dt=1, decimate=1).
  5. Backward-compatible module API (n_params, _init_worker, _problem, _screen_task,
     run_lm, lyapunov, available_cpus) for jacobian_sweep.py.

Examples:
    # 1. Process an entire folder with a fixed (L, m, q) and automatic dt/decimate
    python main_jacobian_v4.py --input deter_test_sweep --key x --auto-dt \
        --lags 2 --ms 4 --qs 10 --out batch_results

    # 2. Process only chaotic/deterministic files matching a glob pattern
    python main_jacobian_v4.py --input deter_test_sweep --pattern "DET_*.mat" \
        --key x --auto-dt --lags 2 --ms 4 --qs 10 --out det_results.json

    # 3. Single file with explicit dt and decimate
    python main_jacobian_v4.py --input deter_test_sweep/DET_lorenz_x_r02.mat \
        --key x --dt 0.02 --decimate 5 --lags 2 --ms 4 --qs 10
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
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
import scipy.io

# Worker-global state (shared interface with jacobian_sweep.py)
_G = {}


def available_cpus():
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm))
    return max(1, os.cpu_count() or 1)


def n_params(q, m):
    """Number of parameters in a 1-hidden-layer network with m inputs and q hidden units."""
    return 1 + q * (m + 2)


# ----------------------------------------------------------------------------
# Data loading, parameter inference & delay embedding
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


def infer_dt_decimate(filepath, n_raw):
    """Infer benchmark (dt, decimate) from standard signal names in the filename."""
    name = os.path.basename(filepath).lower()
    if "lorenz" in name:
        return 0.02, 5
    if "rossler" in name:
        return 0.1, 5
    if "linear" in name or "exponential" in name:
        return 1.0 / max(n_raw - 1, 1), 1
    if "quadratic" in name or "polynomial" in name:
        return 2.0 / max(n_raw - 1, 1), 1
    return 1.0, 1


def _problem(L, m, q, x=None):
    """Build delay-embedding matrices (X, y) for map X_t -> X_{t+L}."""
    if x is None:
        x = _G["x"]
    N = len(x)
    T = N - m * L
    if T <= 1:
        raise ValueError(f"Series too short (N={N}) for m={m}, L={L}")
    # Row t: [x_{t+(m-1)L}, x_{t+(m-2)L}, ..., x_t] (newest first), target y_t = x_{t+mL}
    X = np.column_stack([x[(m - 1 - i) * L : (m - 1 - i) * L + T] for i in range(m)])
    y = x[m * L : m * L + T]
    return SimpleNamespace(L=L, m=m, q=q, X=X, y=y, T=T)


# ----------------------------------------------------------------------------
# Neural network forward pass, parameter Jacobian, and Levenberg-Marquardt
# ----------------------------------------------------------------------------
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
    """Compute residual r = y_hat - y and analytic parameter Jacobian J_th."""
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
    G = dH * w  # (T, q)
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
    """Pick the best random-feature initialization among n_cand candidates."""
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


def run_lm(prob, th0, max_nfev, tol):
    """Damped Levenberg-Marquardt nonlinear least-squares solver with analytic Jacobian."""
    fn = _G["cfg"]["activation"]
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


# ----------------------------------------------------------------------------
# Lyapunov exponent estimators (raw per-step + orbit-consistent per-time-unit)
# ----------------------------------------------------------------------------
def lyapunov(X, th, q, fn, stride=1):
    """
    Largest Lyapunov exponent PER MAP STEP along X[stride-1::stride] via QR
    decomposition of companion Jacobians.
    Kept un-normalized by L for compatibility with jacobian_sweep.py.
    """
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


def lyapunov_orb(X, th, q, fn, L, tau=1.0):
    """
    Orbit-consistent largest Lyapunov exponent PER TIME UNIT:
      lambda = (1 / (L * tau)) * mean_{r=0..L-1} lyapunov(X[r:], th, q, fn, stride=L)
    """
    if L == 1:
        lam_step = lyapunov(X, th, q, fn, stride=1)
    else:
        vals = [lyapunov(X[r:], th, q, fn, stride=L) for r in range(L)]
        if not all(np.isfinite(vals)):
            return float("nan")
        lam_step = float(np.mean(vals))
    return lam_step / (L * tau)


# ----------------------------------------------------------------------------
# Parallel screening & polishing workers
# ----------------------------------------------------------------------------
def _init_worker(x, fit_cfg):
    _G["x"] = x
    _G["cfg"] = fit_cfg


def _screen_task(task):
    L, m, q, st = task
    cfg = _G["cfg"]
    prob = _problem(L, m, q)
    ss = np.random.SeedSequence([cfg["seed"], L, m, q, st])
    rng = np.random.default_rng(ss)
    th0 = _rf_init(prob, rng, cfg["rf_cand"], cfg["activation"])
    th, sse = run_lm(prob, th0, cfg["screen_nfev"], cfg["screen_tol"])
    return L, m, q, st, sse, th


def _polish_task(task):
    L, m, q, rank, th0 = task
    cfg = _G["cfg"]
    prob = _problem(L, m, q)
    th, sse = run_lm(prob, th0, cfg["polish_nfev"], cfg["polish_tol"])
    lam_orb = float("nan")
    if np.isfinite(sse):
        lam_orb = lyapunov_orb(
            prob.X, th, q, cfg["activation"], L, tau=cfg.get("tau", 1.0)
        )
    return L, m, q, rank, sse, lam_orb


# ----------------------------------------------------------------------------
# Grid search, BenSaida selection & single-file pipeline
# ----------------------------------------------------------------------------
def build_grid(N, lags, ms, qs):
    combos = [
        (L, m, q)
        for L in lags
        for m in ms
        for q in qs
        if N - m * L > n_params(q, m) + 1
    ]
    combos.sort(key=lambda c: -n_params(c[2], c[1]))
    return combos


def fit_and_estimate(x, combos, fit_cfg, workers, starts, topk, quiet=False):
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(x, fit_cfg)
    ) as pool:
        stasks = [(L, m, q, st) for (L, m, q) in combos for st in range(starts)]
        screened = {c: [] for c in combos}
        done = {c: 0 for c in combos}
        n_done = 0
        for fut in as_completed([pool.submit(_screen_task, t) for t in stasks]):
            L, m, q, _, sse, th = fut.result()
            c = (L, m, q)
            screened[c].append((sse, th))
            done[c] += 1
            if done[c] == starts:
                n_done += 1
                if not quiet and (
                    n_done % max(1, len(combos) // 10) == 0 or n_done == len(combos)
                ):
                    print(
                        f"  screen {n_done:4d}/{len(combos)} combos  [{time.time() - t0:6.1f}s]",
                        flush=True,
                    )

        ptasks = []
        for c, lst in screened.items():
            lst = sorted((v for v in lst if np.isfinite(v[0])), key=lambda v: v[0])
            for rank, (_, th) in enumerate(lst[:topk]):
                ptasks.append((*c, rank, th))
        ptasks.sort(key=lambda t: -n_params(t[2], t[1]))

        polished = {c: [] for c in combos}
        for i, fut in enumerate(
            as_completed([pool.submit(_polish_task, t) for t in ptasks]), 1
        ):
            L, m, q, _, sse, lam_orb = fut.result()
            polished[(L, m, q)].append((sse, lam_orb))
            if not quiet and (i % max(1, len(ptasks) // 10) == 0 or i == len(ptasks)):
                print(
                    f"  polish {i:4d}/{len(ptasks)} runs    [{time.time() - t0:6.1f}s]",
                    flush=True,
                )

    return polished


def select_bensaida(rows, tau, max_spread):
    """Largest lambda_orb among models whose spread across polished starts (per sample) <= max_spread."""
    ok = [
        r
        for r in rows
        if np.isfinite(r["lambda"])
        and np.isfinite(r["spread"])
        and (r["spread"] * tau <= max_spread)
    ]
    if not ok:
        ok = [r for r in rows if np.isfinite(r["lambda"])]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["lambda"], -(r["L"] + r["m"] + r["q"])))


def process_one_file(filepath, args, lags, ms, qs, quiet=False):
    """Load one file, run grid search, and return its full result dictionary."""
    x_raw = load_series(filepath, key=args.key, column=args.column)
    n_raw = len(x_raw)

    if args.auto_dt:
        inferred_dt, inferred_dec = infer_dt_decimate(filepath, n_raw)
    else:
        inferred_dt, inferred_dec = 1.0, 1

    dt = float(args.dt) if args.dt is not None else inferred_dt
    d = max(1, int(args.decimate)) if args.decimate is not None else inferred_dec

    x = x_raw[::d]
    std = float(np.std(x))
    if std == 0.0 or not np.isfinite(std):
        raise ValueError(f"{filepath}: series is constant (std=0)")
    x = (x - np.mean(x)) / std
    N = len(x)
    tau = dt * d

    combos = build_grid(N, lags, ms, qs)
    if not combos:
        raise ValueError(
            f"{filepath}: no valid (L, m, q) combinations fit N={N} points."
        )

    if not quiet:
        print(
            f"Input: {filepath} | N={N} (decimate={d}) | dt={dt:g} | tau={tau:g} | "
            f"grid={len(combos)} combos",
            flush=True,
        )

    fit_cfg = dict(
        activation=args.activation,
        seed=args.seed,
        rf_cand=args.rf_cand,
        screen_nfev=args.screen_nfev,
        screen_tol=args.screen_tol,
        polish_nfev=args.polish_nfev,
        polish_tol=args.polish_tol,
        tau=tau,
    )

    polished = fit_and_estimate(
        x, combos, fit_cfg, args.workers, args.starts, args.topk, quiet=quiet
    )

    var = float(np.var(x))
    rows = []
    for L, m, q in sorted(combos):
        lst = sorted(
            (v for v in polished[(L, m, q)] if np.isfinite(v[0])), key=lambda v: v[0]
        )
        if not lst:
            continue
        sse, lam = lst[0]
        vals = [v[1] for v in lst if np.isfinite(v[1])]
        spread = (max(vals) - min(vals)) if vals else float("nan")
        rows.append(
            {
                "L": L,
                "m": m,
                "q": q,
                "n_par": n_params(q, m),
                "sse": sse,
                "nmse": sse / ((N - m * L) * var),
                "lambda_orb": lam,
                "lambda_step": lam * L * tau,
                "spread": spread,
                "lambda": lam,
            }
        )

    best = select_bensaida(rows, tau, args.max_lam_spread)
    return {
        "input": filepath,
        "file": os.path.basename(filepath),
        "N": N,
        "dt": dt,
        "decimate": d,
        "tau": tau,
        "selected": best,
        "rows": rows,
    }


def collect_mat_files(folder, pattern="*.mat", recursive=False):
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
    matches.sort()
    return matches


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input file (.mat, .npy, .csv, .txt) OR folder containing .mat files",
    )
    p.add_argument(
        "--pattern",
        default="*.mat",
        help="Filename glob filter when --input is a folder (default '*.mat')",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively when --input is a folder",
    )
    p.add_argument("--key", default=None, help="Variable name inside .mat file")
    p.add_argument(
        "--column", type=int, default=0, help="Column index if 2-D array (default 0)"
    )
    p.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampling interval of input series (default 1.0 unless --auto-dt is set)",
    )
    p.add_argument(
        "--decimate",
        type=int,
        default=None,
        help="Keep every d-th sample (default 1 unless --auto-dt is set)",
    )
    p.add_argument(
        "--auto-dt",
        action="store_true",
        help="Infer dt and decimate automatically from benchmark signal names (lorenz_x, rossler_x, etc.)",
    )

    g = p.add_argument_group("hyperparameter grid")
    g.add_argument(
        "--max-l", type=int, default=2, help="Max lag L if --lags not given (default 2)"
    )
    g.add_argument(
        "--max-m",
        type=int,
        default=4,
        help="Max embedding dim m if --ms not given (default 4)",
    )
    g.add_argument(
        "--max-q",
        type=int,
        default=10,
        help="Max hidden units q if --qs not given (default 10)",
    )
    g.add_argument(
        "--lags", type=int, nargs="+", default=None, help="Explicit list of L values"
    )
    g.add_argument(
        "--ms", type=int, nargs="+", default=None, help="Explicit list of m values"
    )
    g.add_argument(
        "--qs", type=int, nargs="+", default=None, help="Explicit list of q values"
    )
    g.add_argument(
        "--activation", choices=["tanh", "logistic", "sigmoid"], default="tanh"
    )

    o = p.add_argument_group("optimization & selection")
    o.add_argument(
        "--starts", type=int, default=10, help="Random starts per (L,m,q) (default 10)"
    )
    o.add_argument(
        "--rf-cand",
        type=int,
        default=30,
        help="Random-feature candidates per start (default 30)",
    )
    o.add_argument(
        "--screen-nfev",
        type=int,
        default=300,
        help="Max LM evals in screening (default 300)",
    )
    o.add_argument("--screen-tol", type=float, default=1e-10)
    o.add_argument(
        "--topk", type=int, default=2, help="Top starts polished per combo (default 2)"
    )
    o.add_argument(
        "--polish-nfev",
        type=int,
        default=2000,
        help="Max LM evals in polishing (default 2000)",
    )
    o.add_argument("--polish-tol", type=float, default=1e-12)
    o.add_argument(
        "--max-lam-spread",
        type=float,
        default=0.1,
        help="Max allowed spread across polished starts (per used sample, default 0.1)",
    )
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--workers", type=int, default=available_cpus())
    o.add_argument(
        "--out",
        default=None,
        help="Output JSON file (single-file or batch summary) OR output directory for batch mode",
    )
    o.add_argument(
        "--resume",
        action="store_true",
        help="In batch directory mode, skip files whose <stem>.json already exists in --out",
    )
    return p.parse_args()


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def run_single(filepath, args, lags, ms, qs):
    payload = process_one_file(filepath, args, lags, ms, qs, quiet=False)
    rows = payload["rows"]
    best = payload["selected"]

    print("\n L  m  q |      NMSE | lambda (orb) | lam/step |  spread")
    print("-" * 58)
    for r in sorted(rows, key=lambda r: r["nmse"]):
        print(
            f"{r['L']:>2} {r['m']:>2} {r['q']:>2} | {r['nmse']:9.2e} | "
            f"{r['lambda_orb']:+12.5f} | {r['lambda_step']:+8.5f} | {r['spread']:7.4f}"
        )

    print("-" * 58)
    if best is not None:
        print(
            f"Selected (BenSaida rule): L={best['L']}, m={best['m']}, q={best['q']} -> "
            f"lambda = {best['lambda_orb']:+.6f} per time unit "
            f"(NMSE={best['nmse']:.2e}, spread={best['spread']:.4f})"
        )
    else:
        print("No valid finite Lyapunov estimate produced.")

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"Saved results to {args.out}")


def run_batch(folder, args, lags, ms, qs):
    files = collect_mat_files(folder, pattern=args.pattern, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No files matching '{args.pattern}' found in '{folder}'.")

    out_is_json = bool(args.out and args.out.lower().endswith(".json"))
    out_dir = None
    if args.out and not out_is_json:
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)

    print(
        f"Batch mode: {len(files)} file(s) in '{folder}' (pattern='{args.pattern}')",
        flush=True,
    )
    print("=" * 88, flush=True)

    t_batch = time.time()
    all_payloads = []
    summary_rows = []

    for idx, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        stem = os.path.splitext(fname)[0]
        per_file_json = os.path.join(out_dir, f"{stem}.json") if out_dir else None

        if args.resume and per_file_json and os.path.exists(per_file_json):
            with open(per_file_json) as f:
                payload = json.load(f)
            best = payload.get("selected")
            lam_str = f"{best['lambda_orb']:+10.5f}" if best else "       nan"
            print(
                f"[{idx:3d}/{len(files)}] {fname:<28} | RESUMED | LE: {lam_str}",
                flush=True,
            )
            all_payloads.append(payload)
            summary_rows.append(_make_summary_row(payload))
            continue

        t0 = time.time()
        try:
            payload = process_one_file(fp, args, lags, ms, qs, quiet=True)
            best = payload["selected"]
            all_payloads.append(payload)
            srow = _make_summary_row(payload)
            summary_rows.append(srow)

            if per_file_json:
                with open(per_file_json, "w") as f:
                    json.dump(payload, f, indent=2, default=_json_default)

            if best is not None:
                print(
                    f"[{idx:3d}/{len(files)}] {fname:<28} | tau={payload['tau']:<7g} | "
                    f"({best['L']},{best['m']},{best['q']:>2}) | "
                    f"NMSE={best['nmse']:.2e} | LE={best['lambda_orb']:+10.5f} | "
                    f"spread={best['spread']:.4f} [{time.time() - t0:.1f}s]",
                    flush=True,
                )
            else:
                print(
                    f"[{idx:3d}/{len(files)}] {fname:<28} | NO VALID ESTIMATE [{time.time() - t0:.1f}s]",
                    flush=True,
                )
        except Exception as exc:
            print(f"[{idx:3d}/{len(files)}] {fname:<28} | ERROR: {exc}", flush=True)
            summary_rows.append(
                {
                    "file": fname,
                    "N": None,
                    "dt": None,
                    "decimate": None,
                    "tau": None,
                    "L": None,
                    "m": None,
                    "q": None,
                    "nmse": float("nan"),
                    "lambda_orb": float("nan"),
                    "spread": float("nan"),
                    "error": str(exc),
                }
            )

    print("=" * 88)
    print(f"Batch finished: {len(files)} files in {time.time() - t_batch:.1f}s")

    if args.out:
        if out_is_json:
            parent = os.path.dirname(os.path.abspath(args.out))
            os.makedirs(parent, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(
                    {
                        "folder": folder,
                        "summary": summary_rows,
                        "results": all_payloads,
                    },
                    f,
                    indent=2,
                    default=_json_default,
                )
            print(f"Saved batch JSON to {args.out}")
        else:
            sum_json = os.path.join(out_dir, "batch_summary.json")
            sum_csv = os.path.join(out_dir, "batch_summary.csv")
            with open(sum_json, "w") as f:
                json.dump(summary_rows, f, indent=2, default=_json_default)
            if summary_rows:
                with open(sum_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                    w.writeheader()
                    w.writerows(summary_rows)
            print(f"Saved per-file JSONs, {sum_json}, and {sum_csv}")


def _make_summary_row(payload):
    best = payload.get("selected")
    return {
        "file": payload["file"],
        "N": payload["N"],
        "dt": payload["dt"],
        "decimate": payload["decimate"],
        "tau": payload["tau"],
        "L": best["L"] if best else None,
        "m": best["m"] if best else None,
        "q": best["q"] if best else None,
        "nmse": best["nmse"] if best else float("nan"),
        "lambda_orb": best["lambda_orb"] if best else float("nan"),
        "spread": best["spread"] if best else float("nan"),
        "error": "",
    }


def main():
    args = parse_args()
    lags = args.lags if args.lags is not None else list(range(1, args.max_l + 1))
    ms = args.ms if args.ms is not None else list(range(1, args.max_m + 1))
    qs = args.qs if args.qs is not None else list(range(1, args.max_q + 1))

    if os.path.isdir(args.input):
        run_batch(args.input, args, lags, ms, qs)
    elif os.path.isfile(args.input):
        run_single(args.input, args, lags, ms, qs)
    else:
        raise SystemExit(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
