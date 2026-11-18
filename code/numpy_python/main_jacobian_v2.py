#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_jacobian.py  --  Lyapunov exponent of a time series with the Jacobian
method (BenSaida / Shintani-Linton), neural-network map fitted by
Levenberg-Marquardt (non-linear least squares), CPU + multiprocessing.
"""

# --- keep every worker single-threaded: must be set BEFORE importing numpy ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.io
from scipy.optimize import least_squares


# ----------------------------------------------------------------------------
# Model: x_t = D + sum_j A_j * psi(C_j + sum_i B_ji * x_{t-iL})
# theta = [B (q*m, row-major), C (q), A (q), D (1)]
# ----------------------------------------------------------------------------
def lag_matrix(x, L, m):
    """Row t -> (x_{t-L}, ..., x_{t-mL}) for t = mL .. N-1  (shape (N-mL, m))."""
    N = len(x)
    return np.column_stack([x[m * L - i * L: N - i * L] for i in range(1, m + 1)])


def act(u, fn):
    """Return (psi(u), psi'(u))."""
    if fn == "tanh":
        p = np.tanh(u)
        return p, 1.0 - p * p
    if fn == "logistic":
        p = 1.0 / (1.0 + np.exp(-u))
        return p, p * (1.0 - p)
    if fn == "sigmoid":
        a = np.abs(u)
        p = u * (1.0 + a / 2.0) / (2.0 + a + u * u / 2.0)
        d = 8.0 * (1.0 + a) / (3.0 + (1.0 + a) ** 2) ** 2
        return p, d
    raise ValueError(f"unknown activation '{fn}'")


def n_params(q, m):
    return q * m + 2 * q + 1


def unpack(th, q, m):
    B = th[:q * m].reshape(q, m)
    C = th[q * m:q * m + q]
    A = th[q * m + q:q * m + 2 * q]
    D = th[q * m + 2 * q]
    return B, C, A, D


class Problem:
    """Residual r = target - y_fit and its analytic Jacobian, with the forward
    pass cached so scipy's separate fun/jac calls share one evaluation."""

    def __init__(self, XLAG, target, q, fn):
        self.X, self.y, self.q, self.fn = XLAG, target, q, fn
        self.T, self.m = XLAG.shape
        self.P = n_params(q, self.m)
        self._key = None

    def _forward(self, th):
        if self._key is not None and np.array_equal(th, self._key):
            return
        B, C, A, D = unpack(th, self.q, self.m)
        p, dp = act(self.X @ B.T + C, self.fn)
        self._p, self._dp = p, dp
        self._res = self.y - (D + p @ A)
        self._key = th.copy()

    def residual(self, th):
        self._forward(th)
        return self._res

    def jacobian(self, th):
        self._forward(th)
        q, m, T = self.q, self.m, self.T
        A = th[q * m + q:q * m + 2 * q]
        dydC = self._dp * A                                           # (T, q)
        J = np.empty((T, self.P))
        J[:, :q * m] = (dydC[:, :, None] * self.X[:, None, :]).reshape(T, q * m)
        J[:, q * m:q * m + q] = dydC
        J[:, q * m + q:q * m + 2 * q] = self._p
        J[:, -1] = 1.0
        np.negative(J, out=J)                                         # residual = target - y
        return J


def run_lm(prob, th0, max_nfev, tol):
    """One Levenberg-Marquardt run. Returns (theta, SSE) -- SSE = inf on failure."""
    try:
        with np.errstate(all="ignore"):
            r = least_squares(prob.residual, th0, jac=prob.jacobian, method="lm",
                              xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)
        sse = float(np.sum(r.fun ** 2))
        if not np.isfinite(sse):
            return th0, math.inf
        return r.x, sse
    except Exception:
        return th0, math.inf


# ----------------------------------------------------------------------------
# Initialisation: random hidden layers, then (A, D) by linear least squares
# ----------------------------------------------------------------------------
def init_theta(prob, rng, n_cand):
    q, m = prob.q, prob.m
    best_sse, best = math.inf, None
    for c in range(max(1, n_cand)):
        s = 0.5 if n_cand <= 0 else math.exp(rng.uniform(math.log(0.3), math.log(5.0)))
        B = rng.standard_normal((q, m)) * s
        C = rng.standard_normal(q) * s
        if n_cand <= 0:                       # plain random start
            A = rng.standard_normal(q) / math.sqrt(q)
            return np.concatenate([B.ravel(), C, A, [0.0]])
        F = np.hstack([act(prob.X @ B.T + C, prob.fn)[0], np.ones((prob.T, 1))])
        w, *_ = np.linalg.lstsq(F, prob.y, rcond=None)
        sse = float(np.sum((prob.y - F @ w) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = np.concatenate([B.ravel(), C, w[:q], [w[q]]])
    return best


# ----------------------------------------------------------------------------
# Lyapunov exponent: renormalised product of companion Jacobians
# ----------------------------------------------------------------------------
def lyapunov(XLAG, th, q, fn, stride):
    """lambda = (1/M) sum_t log || J_t v_{t-1} ||, v_0 = e_1, v renormalised each step."""
    T, m = XLAG.shape
    B, C, A, _ = unpack(th, q, m)
    _, dp = act(XLAG @ B.T + C, fn)
    dy = (dp * A) @ B                                   # (T, m): d f / d x_{t-iL}
    idx = np.arange(stride - 1, T, stride)
    if len(idx) == 0:
        return float("nan")
    if m == 1:
        a = np.abs(dy[idx, 0])
        if np.any(a < 1e-300):                          # saturated network: zero derivative
            return -math.inf
        return float(np.mean(np.log(a)))
    v = np.zeros(m)
    v[0] = 1.0
    acc = 0.0
    nv = np.empty(m)
    for t in idx:
        nv[0] = dy[t] @ v
        nv[1:] = v[:-1]
        n = math.sqrt(float(nv @ nv))
        if n < 1e-300:
            return -math.inf
        acc += math.log(n)
        v = nv / n
    return acc / len(idx)


# ----------------------------------------------------------------------------
# Worker side
# ----------------------------------------------------------------------------
_G = {}


def _init_worker(x, cfg):
    _G["x"], _G["cfg"] = x, cfg
    _G["lag"], _G["prob"] = {}, {}


def _problem(L, m, q):
    key = (L, m, q)
    if key not in _G["prob"]:
        if (L, m) not in _G["lag"]:
            _G["lag"][(L, m)] = (lag_matrix(_G["x"], L, m), _G["x"][m * L:])
        XLAG, target = _G["lag"][(L, m)]
        _G["prob"][key] = Problem(XLAG, target, q, _G["cfg"]["activation"])
    return _G["prob"][key]


def _screen_task(args):
    L, m, q, start = args
    cfg = _G["cfg"]
    prob = _problem(L, m, q)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], L, m, q, start]))
    th0 = init_theta(prob, rng, cfg["rf_cand"])
    th, sse = run_lm(prob, th0, cfg["screen_nfev"], cfg["screen_tol"])
    return L, m, q, start, sse, th


def _polish_task(args):
    L, m, q, rank, th0 = args
    cfg = _G["cfg"]
    prob = _problem(L, m, q)
    th, sse = run_lm(prob, th0, cfg["polish_nfev"], cfg["polish_tol"])
    lam_T = lam_h = float("nan")
    if np.isfinite(sse):
        h = max(1, int(round(prob.T ** (1.0 / 3.0))))
        lam_T = lyapunov(prob.X, th, q, cfg["activation"], 1)
        lam_h = lyapunov(prob.X, th, q, cfg["activation"], h)
    return L, m, q, rank, sse, lam_T, lam_h, th


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def logistic_map(r, n, burn=1000, x0=0.4):
    x = np.empty(n + burn)
    x[0] = x0
    for i in range(1, n + burn):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x[burn:]


def logistic_reference_lambda(r, n=200000):
    x = logistic_map(r, n)
    return float(np.mean(np.log(np.abs(r * (1.0 - 2.0 * x)))))


def load_signals(args):
    """Return list of (name, 1-D float64 array)."""
    if args.signal == "logistic":
        return [(f"logistic_r{args.r}", logistic_map(args.r, args.n))]
    if args.signal == "white":
        rng = np.random.default_rng(args.seed)
        return [("white_noise", rng.standard_normal(args.n))]
    path = args.input
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        data = np.asarray(scipy.io.loadmat(path)[args.key]).squeeze()
    elif ext == ".npy":
        data = np.load(path)
    else:
        data = np.loadtxt(path, delimiter=args.delimiter)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return [(os.path.basename(path), data)]
    if data.shape[0] < data.shape[1]:
        data = data.T
    if args.imf == "all":
        ids = range(data.shape[1])
    else:
        ids = [int(i) for i in args.imf.split(",")]
    return [(f"imf{i}", data[:, i].copy()) for i in ids]


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def fmt(v, spec):
    if np.isfinite(v):
        return format(v, spec)
    return "    -inf" if v == -math.inf else ("    +inf" if v == math.inf else "     nan")


def analyse(x_raw, name, cfg, workers):
    d = cfg["decimate"]
    x = x_raw[::d] if d > 1 else x_raw
    x = (x - x.mean()) / x.std()
    N = len(x)
    lags = cfg["lags"] or list(range(1, cfg["max_l"] + 1))
    combos = [(L, m, q) for L in lags for m in range(1, cfg["max_m"] + 1)
              for q in range(1, cfg["max_q"] + 1)
              if N - m * L > n_params(q, m) + 1]
    combos.sort(key=lambda c: -n_params(c[2], c[1]))
    n_starts = cfg["starts"]
    print(f"\n=== {name}: N={N}{' (decimated x%d)' % d if d > 1 else ''}, "
          f"{len(combos)} combos x {n_starts} starts, {workers} workers ===", flush=True)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(x, cfg)) as pool:
        # ---- pass 1: screening ------------------------------------------------
        tasks = [(L, m, q, s) for (L, m, q) in combos for s in range(n_starts)]
        screened = {c: [] for c in combos}
        done_combo = {c: 0 for c in combos}
        n_done_combos = 0
        for fut in as_completed([pool.submit(_screen_task, t) for t in tasks]):
            L, m, q, s, sse, th = fut.result()
            c = (L, m, q)
            screened[c].append((sse, th))
            done_combo[c] += 1
            if done_combo[c] == n_starts:
                n_done_combos += 1
                best = min(v[0] for v in screened[c])
                print(f"  screen [{n_done_combos:3d}/{len(combos)}] (L={L}, m={m}, q={q}) "
                      f"best SSE={best:.3e}  [{time.time() - t0:6.1f}s]", flush=True)
        # ---- pass 2: polish top-k --------------------------------------------
        ptasks = []
        for c, lst in screened.items():
            lst = [v for v in lst if np.isfinite(v[0])]
            lst.sort(key=lambda v: v[0])
            for rank, (_, th) in enumerate(lst[:cfg["topk"]]):
                ptasks.append((*c, rank, th))
        ptasks.sort(key=lambda t: -n_params(t[2], t[1]))
        polished = {c: [] for c in combos}
        for i, fut in enumerate(as_completed([pool.submit(_polish_task, t) for t in ptasks]), 1):
            L, m, q, rank, sse, lam_T, lam_h, th = fut.result()
            polished[(L, m, q)].append((sse, lam_T, lam_h, th))
            if i % max(1, len(ptasks) // 10) == 0 or i == len(ptasks):
                print(f"  polish [{i:4d}/{len(ptasks)}]  [{time.time() - t0:6.1f}s]", flush=True)

    # ---- table ----------------------------------------------------------------
    rows = []
    for c in sorted(polished):
        lst = [v for v in polished[c] if np.isfinite(v[0])]
        if not lst:
            continue
        lst.sort(key=lambda v: v[0])
        sse, lam_T, lam_h, th = lst[0]
        target = x[c[1] * c[0]:]
        T = len(target)
        var_t = float(np.var(target))
        lamsT = [v[1] for v in lst if not np.isnan(v[1])]
        rows.append(dict(L=c[0], m=c[1], q=c[2], sse=sse, nmse=sse / (T * var_t),
                         rmse=math.sqrt(sse / T),
                         lam_T=lam_T, lam_h=lam_h,
                         lamT_min=min(lamsT) if lamsT else float("nan"),
                         lamT_max=max(lamsT) if lamsT else float("nan"),
                         theta=th))
    if not rows:
        print("  no valid fit.")
        return None
    min_sse = min(r["sse"] for r in rows)

    def stable(r):
        if r["lamT_max"] < -0.5:
            return True
        sp = r["lamT_max"] - r["lamT_min"]
        return bool(np.isfinite(sp)) and sp <= cfg["max_lam_spread"]

    for r in rows:
        r["stable"] = stable(r)
    eligible = [r for r in rows if r["sse"] <= cfg["max_sse_ratio"] * min_sse and r["stable"]]
    if not eligible:
        print("  WARNING: no combo has a stable lambda across its polished starts; "
              "selecting among all combos (treat the result with caution).")
        eligible = [r for r in rows if r["sse"] <= cfg["max_sse_ratio"] * min_sse]
    if cfg["select"] == "sse":
        best = min(eligible, key=lambda r: r["sse"])
    else:
        key = "lam_T" if cfg["select"] == "lambda_T" else "lam_h"
        best = max(eligible, key=lambda r: (r[key] if np.isfinite(r[key]) else -math.inf,
                                            -(r["L"] + r["m"] + r["q"])))

    sc = 1.0 / d
    print(f"\n  {'L':>2} {'m':>2} {'q':>2} | {'SSE':>10} {'NMSE':>9} | {'lam_T':>8} {'lam_h':>8} "
          f"| lam_T over polished starts (min..max)")
    for r in rows:
        mark = "  <== selected" if r is best else ("" if r["stable"] else "  (unstable lambda)")
        print(f"  {r['L']:>2} {r['m']:>2} {r['q']:>2} | {r['sse']:10.3e} {r['nmse']:9.2e} | "
              f"{fmt(r['lam_T'], '+8.4f')} {fmt(r['lam_h'], '+8.4f')} | "
              f"{fmt(r['lamT_min'], '+.4f')} .. {fmt(r['lamT_max'], '+.4f')}{mark}")
    print(f"\n  RESULT {name}: (L,m,q)=({best['L']},{best['m']},{best['q']})  "
          f"lambda_T={best['lam_T'] * sc:+.4f}  lambda_h={best['lam_h'] * sc:+.4f}"
          f"{' (per original sample)' if d > 1 else ''}  "
          f"SSE={best['sse']:.3e}  NMSE={best['nmse']:.2e}  [{time.time() - t0:.1f}s]", flush=True)

    # Reconstruction metrics on standardised signal
    bL, bm, bq = best["L"], best["m"], best["q"]
    XLAG_b = lag_matrix(x, bL, bm)
    y_target = x[bm * bL:]
    B_b, C_b, A_b, D_b = unpack(best["theta"], bq, bm)
    y_fit = D_b + act(XLAG_b @ B_b.T + C_b, cfg["activation"])[0] @ A_b
    diff = y_target - y_fit
    sse_val = float(np.sum(diff ** 2))
    nmse_val = sse_val / (len(y_target) * float(np.var(y_target)))
    rmse_val = float(np.sqrt(np.mean(diff ** 2)))
    mad_val = float(np.mean(np.abs(diff)))
    max_ad_val = float(np.max(np.abs(diff)))

    print(f"\n--- Reconstruction Differences ---")
    print(f"Sum of Squared Errors (SSE):    {sse_val:.6g}")
    print(f"Normalized MSE (NMSE):          {nmse_val:.6g}")
    print(f"Root Mean Squared Error (RMSE): {rmse_val:.6g}")
    print(f"Mean Absolute Difference:       {mad_val:.6g}")
    print(f"Maximum Absolute Difference:    {max_ad_val:.6g}", flush=True)

    out = dict(name=name, N=N, decimate=d, selected=dict(L=best["L"], m=best["m"], q=best["q"]),
               lambda_T=best["lam_T"] * sc, lambda_h=best["lam_h"] * sc,
               sse=sse_val, nmse=nmse_val, rmse=rmse_val,
               mean_abs_diff=mad_val, max_abs_diff=max_ad_val,
               seconds=time.time() - t0,
               table=[{k: v for k, v in r.items() if k != "theta"} for r in rows])
    return out, best, x


def plot_reconstruction(x, best, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L, m, q = best["L"], best["m"], best["q"]
    XLAG = lag_matrix(x, L, m)
    B, C, A, D = unpack(best["theta"], q, m)
    y = D + act(XLAG @ B.T + C, cfg["activation"])[0] @ A
    t = np.arange(m * L, len(x))
    k = min(len(t), 1000)
    plt.figure(figsize=(10, 4))
    plt.plot(t[:k], x[m * L:][:k], "k", alpha=.8, label="signal (standardised)")
    plt.plot(t[:k], y[:k], "r--", label=f"fit (L={L}, m={m}, q={q})")
    plt.legend(); plt.grid(ls=":", alpha=.6); plt.xlabel("t"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_argument_group("signal")
    s.add_argument("--signal", choices=["file", "logistic", "white"], default="file")
    s.add_argument("--input", help=".mat / .npy / .csv (2-D: columns = IMFs, auto-transposed if rows < cols)")
    s.add_argument("--key", default="x", help="variable name inside a .mat file")
    s.add_argument("--imf", default="0", help="'all' or comma list of column indices (0-based)")
    s.add_argument("--delimiter", default=",")
    s.add_argument("--n", type=int, default=10000, help="length for synthetic signals")
    s.add_argument("--r", type=float, default=3.9, help="logistic-map parameter")
    s.add_argument("--decimate", type=int, default=1,
                   help="keep every d-th sample; lambda is reported per ORIGINAL sample (divided by d)")
    g = p.add_argument_group("model grid")
    g.add_argument("--max-l", type=int, default=2)
    g.add_argument("--max-m", type=int, default=3)
    g.add_argument("--max-q", type=int, default=6)
    g.add_argument("--lags", type=int, nargs="*", default=None, help="explicit list of L values (overrides --max-l)")
    g.add_argument("--activation", choices=["tanh", "logistic", "sigmoid"], default="tanh")
    o = p.add_argument_group("optimisation")
    o.add_argument("--starts", type=int, default=20, help="random starts per (L,m,q)")
    o.add_argument("--rf-cand", type=int, default=30,
                   help="random-feature candidates per start (0 = plain random start)")
    o.add_argument("--screen-nfev", type=int, default=300)
    o.add_argument("--screen-tol", type=float, default=1e-10)
    o.add_argument("--topk", type=int, default=3, help="starts polished per (L,m,q)")
    o.add_argument("--polish-nfev", type=int, default=3000)
    o.add_argument("--polish-tol", type=float, default=1e-12)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--workers", type=int, default=available_cpus())
    r = p.add_argument_group("selection / output")
    r.add_argument("--select", choices=["lambda_T", "lambda_h", "sse"], default="lambda_T")
    r.add_argument("--max-lam-spread", type=float, default=0.1,
                   help="a combo is eligible only if lambda_T varies by at most this much across its "
                        "polished starts (inf disables). Guards against spurious Jacobians of over-embedded fits")
    r.add_argument("--max-sse-ratio", type=float, default=math.inf,
                   help="only combos with SSE <= ratio * best SSE can be selected")
    r.add_argument("--out", default="results_jacobian.json")
    r.add_argument("--plot", action="store_true", help="save a reconstruction PNG per signal")
    return p.parse_args()


def main():
    args = parse()
    cfg = vars(args).copy()
    signals = load_signals(args)
    if args.signal == "logistic":
        print(f"reference lambda (long orbit) for r={args.r}: {logistic_reference_lambda(args.r):+.4f}")
    results = []
    for name, x in signals:
        res = analyse(x, name, cfg, args.workers)
        if res is None:
            continue
        out, best, xs = res
        results.append(out)
        if args.plot:
            plot_reconstruction(xs, best, cfg, f"{os.path.splitext(args.out)[0]}_{name}.png")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
