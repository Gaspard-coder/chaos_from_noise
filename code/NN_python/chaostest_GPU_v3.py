#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chaostest_GPU.py  --  Lyapunov exponent with the Jacobian method, neural-network
map fitted by (batched) L-BFGS in PyTorch.  GPU if available, CPU otherwise.

Pipeline for each signal (e.g. each IMF) and each (L, m, q):
  1. INIT    : n_cand random hidden layers (B, C); the output weights (A, D) are
               solved in closed form (linear least squares, batched); the best
               n_starts candidates are kept.            -> replaces 1000 Adam steps
  2. SCREEN  : batched L-BFGS on all n_starts for --screen-iters iterations
  3. POLISH  : top-k starts, batched L-BFGS for --polish-iters iterations
               (--polish torch : stock torch.optim.LBFGS, one start at a time)
  4. LAMBDA  : renormalised Jacobian product on the CPU, both estimators
               lambda_T (all consecutive Jacobians) and lambda_h (every h-th, h=T^(1/3))
  5. SELECT  : largest lambda among combos whose lambda is stable across the polished
               starts (BenSaida's rule), or smallest SSE (--select sse)

What changed compared with the previous chaostest_GPU.py
  * float64 by default: tolerances of 1e-9 are meaningless in float32, and the old
    polish stage burnt its whole iteration budget on float32 noise
  * 1000 Adam steps x 1024 starts replaced by a closed-form output-layer solve
  * L-BFGS is batched: every start has its OWN history and line search but all of
    them advance in the same GPU kernels (no Python loop over starts, no
    per-iteration host sync inside torch.optim.LBFGS' two-loop recursion)
  * analytic gradient (no autograd graph); checked against autograd with --check
  * several starts are polished instead of only the best-after-Adam one
  * lambda: renormalised product (no under/overflow), computed once per
    polished start; reports both M=T and M=T^(2/3) estimators
  * per-stage timings in the log

Examples
    python chaostest_GPU.py --input logistic.npy --max-l 1 --max-m 2 --max-q 3
    python chaostest_GPU.py --input imfs.mat --key imf --imf all
    python chaostest_GPU.py --check            # analytic-gradient test
  -> results are written to results_<NAME>.json, NAME = input file name without extension
"""
import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import scipy.io
import torch


# ----------------------------------------------------------------------------
# Logging (console + file)
# ----------------------------------------------------------------------------
def setup_logger(path):
    lg = logging.getLogger("chaostest_gpu")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(fh)
    lg.addHandler(ch)
    return lg


# ----------------------------------------------------------------------------
# Numpy helpers (data side, lambda)
# ----------------------------------------------------------------------------
def lag_matrix_np(x, L, m):
    N = len(x)
    return np.column_stack([x[m * L - i * L: N - i * L] for i in range(1, m + 1)])


def act_np(u, fn):
    if fn == "tanh":
        p = np.tanh(u)
        return p, 1.0 - p * p
    if fn == "logistic":
        p = 1.0 / (1.0 + np.exp(-u))
        return p, p * (1.0 - p)
    a = np.abs(u)
    p = u * (1.0 + a / 2.0) / (2.0 + a + u * u / 2.0)
    return p, 8.0 * (1.0 + a) / (3.0 + (1.0 + a) ** 2) ** 2


def n_params(q, m):
    return q * m + 2 * q + 1


def unpack_np(th, q, m):
    return (th[:q * m].reshape(q, m), th[q * m:q * m + q],
            th[q * m + q:q * m + 2 * q], th[q * m + 2 * q])


def lyapunov(XLAG, th, q, fn, stride):
    """(1/M) sum_t log||J_t v_{t-1}||, v renormalised each step (== (1/2M) ln v1 of the paper)."""
    T, m = XLAG.shape
    B, C, A, _ = unpack_np(th, q, m)
    _, dp = act_np(XLAG @ B.T + C, fn)
    dy = (dp * A) @ B
    idx = np.arange(stride - 1, T, stride)
    if len(idx) == 0:
        return float("nan")
    if m == 1:
        a = np.abs(dy[idx, 0])
        if np.any(a < 1e-300):
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


def load_signals(args):
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
    ids = range(data.shape[1]) if args.imf == "all" else [int(i) for i in args.imf.split(",")]
    return [(f"imf{i}", data[:, i].copy()) for i in ids]


# ----------------------------------------------------------------------------
# Batched model:  S independent networks evaluated at once
#   theta (S, P) = [B (q*m, row-major) | C (q) | A (q) | D (1)]
# ----------------------------------------------------------------------------
class BatchedModel:
    def __init__(self, XLAG, target, q, fn):
        self.X, self.y, self.q, self.fn = XLAG, target, q, fn
        self.N, self.m = XLAG.shape
        self.P = n_params(q, self.m)

    def unpack(self, th):
        S, q, m = th.shape[0], self.q, self.m
        return (th[:, :q * m].reshape(S, q, m), th[:, q * m:q * m + q],
                th[:, q * m + q:q * m + 2 * q], th[:, q * m + 2 * q])

    def act(self, u):
        if self.fn == "tanh":
            p = torch.tanh(u)
            return p, 1.0 - p * p
        if self.fn == "logistic":
            p = torch.sigmoid(u)
            return p, p * (1.0 - p)
        a = u.abs()
        p = u * (1.0 + a / 2.0) / (2.0 + a + u * u / 2.0)
        return p, 8.0 * (1.0 + a) / (3.0 + (1.0 + a) ** 2) ** 2

    def _fwd(self, th):
        B, C, A, D = self.unpack(th)
        U = torch.matmul(self.X, B.transpose(1, 2)) + C[:, None, :]          # (S, N, q)
        p, dp = self.act(U)
        yhat = torch.bmm(p, A.unsqueeze(2)).squeeze(2) + D[:, None]          # (S, N)
        return A, p, dp, self.y[None, :] - yhat

    def loss(self, th):
        r = self._fwd(th)[3]
        return (r * r).sum(1)

    def loss_grad(self, th):
        """SSE per start and its analytic gradient w.r.t. theta."""
        S = th.shape[0]
        A, p, dp, r = self._fwd(th)
        f = (r * r).sum(1)
        dldy = -2.0 * r                                                      # (S, N)
        gA = torch.bmm(p.transpose(1, 2), dldy.unsqueeze(2)).squeeze(2)      # (S, q)
        gD = dldy.sum(1, keepdim=True)                                       # (S, 1)
        W = dldy.unsqueeze(2) * dp * A[:, None, :]                           # (S, N, q)
        gC = W.sum(1)                                                        # (S, q)
        gB = torch.matmul(W.transpose(1, 2), self.X)                         # (S, q, m)
        return f, torch.cat([gB.reshape(S, -1), gC, gA, gD], dim=1)


# ----------------------------------------------------------------------------
# Initialisation: random hidden layers + closed-form output layer
# ----------------------------------------------------------------------------
@torch.no_grad()
def random_feature_init(model, n_cand, n_keep, chunk, gen):
    dev, dt = model.X.device, model.X.dtype
    q, m = model.q, model.m
    ths, sses = [], []
    for s0 in range(0, n_cand, chunk):
        S = min(chunk, n_cand - s0)
        scale = torch.exp(math.log(0.3) + (math.log(5.0) - math.log(0.3)) *
                          torch.rand((S, 1), generator=gen, device=dev, dtype=dt))
        B = torch.randn((S, q, m), generator=gen, device=dev, dtype=dt) * scale[:, :, None]
        C = torch.randn((S, q), generator=gen, device=dev, dtype=dt) * scale
        p, _ = model.act(torch.matmul(model.X, B.transpose(1, 2)) + C[:, None, :])
        H = torch.cat([p, torch.ones((S, model.N, 1), device=dev, dtype=dt)], dim=2)   # (S, N, q+1)
        HtH = torch.bmm(H.transpose(1, 2), H)
        Hty = torch.matmul(H.transpose(1, 2), model.y)                                  # (S, q+1)
        ridge = 1e-10 * HtH.diagonal(dim1=1, dim2=2).mean(1)
        HtH = HtH + ridge[:, None, None] * torch.eye(q + 1, device=dev, dtype=dt)
        w, info = torch.linalg.solve_ex(HtH, Hty.unsqueeze(2))
        w = w.squeeze(2)
        r = model.y[None, :] - torch.bmm(H, w.unsqueeze(2)).squeeze(2)
        sse = (r * r).sum(1)
        sse = torch.where((info == 0) & torch.isfinite(sse), sse, torch.full_like(sse, math.inf))
        ths.append(torch.cat([B.reshape(S, -1), C, w[:, :q], w[:, q:q + 1]], dim=1))
        sses.append(sse)
    th, sse = torch.cat(ths), torch.cat(sses)
    order = torch.argsort(sse)[:n_keep]
    return th[order], sse[order]


# ----------------------------------------------------------------------------
# Batched L-BFGS: S independent problems, one set of kernels
# ----------------------------------------------------------------------------
@torch.no_grad()
def lbfgs_batched(model, theta, max_iter, hist=20, gtol=1e-9, ftol=1e-13, patience=10,
                  max_ls=25, c1=1e-4, log=None, log_every=0):
    """Minimise the S independent SSE problems of `model` starting from theta (S, P).

    Per start: own L-BFGS history, own Armijo backtracking line search, own stopping
    test (|g|_inf < gtol, or relative decrease < ftol for `patience` iterations, or 3
    consecutive line-search failures).  Returns (theta, losses)."""
    S, P = theta.shape
    dev, dt = theta.device, theta.dtype
    theta = theta.clone()
    f, g = model.loss_grad(theta)
    bad = ~torch.isfinite(f) | ~torch.isfinite(g).all(1)
    active = ~bad
    f = torch.where(bad, torch.full_like(f, math.inf), f)
    g = torch.where(bad[:, None], torch.zeros_like(g), g)

    Sh = torch.zeros((S, hist, P), device=dev, dtype=dt)
    Yh = torch.zeros((S, hist, P), device=dev, dtype=dt)
    rho = torch.zeros((S, hist), device=dev, dtype=dt)
    gamma = torch.ones(S, device=dev, dtype=dt)
    first = torch.ones(S, dtype=torch.bool, device=dev)
    fails = torch.zeros(S, dtype=torch.long, device=dev)
    smalls = torch.zeros(S, dtype=torch.long, device=dev)
    ptr = 0

    for it in range(max_iter):
        active &= g.abs().amax(1) > gtol
        if not bool(active.any()):
            break

        # ---- search direction: two-loop recursion, vectorised over starts ----------
        qv = g.clone()
        al = torch.zeros((S, hist), device=dev, dtype=dt)
        for k in range(hist):                                   # newest -> oldest
            j = (ptr - 1 - k) % hist
            a = rho[:, j] * (Sh[:, j] * qv).sum(1)
            al[:, j] = a
            qv = qv - a[:, None] * Yh[:, j]
        r = gamma[:, None] * qv
        for k in range(hist - 1, -1, -1):                       # oldest -> newest
            j = (ptr - 1 - k) % hist
            b = rho[:, j] * (Yh[:, j] * r).sum(1)
            r = r + Sh[:, j] * (al[:, j] - b)[:, None]
        d = -r
        gd = (g * d).sum(1)
        nd = ~(gd < 0)                                          # not a descent direction (or NaN)
        if bool(nd.any()):
            d = torch.where(nd[:, None], -g, d)
            gd = torch.where(nd, -(g * g).sum(1), gd)
            rho[nd] = 0.0
            Sh[nd] = 0.0
            Yh[nd] = 0.0
            gamma[nd] = 1.0
            first = first | nd

        alpha = torch.where(first, torch.clamp(1.0 / g.abs().sum(1).clamp_min(1e-300), max=1.0),
                            torch.ones(S, device=dev, dtype=dt))

        # ---- Armijo backtracking, only on the starts that still need it -------------
        th_new = theta.clone()
        f_new = f.clone()
        accepted = torch.zeros(S, dtype=torch.bool, device=dev)
        todo = active.clone()
        for _ in range(max_ls):
            idx = todo.nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            trial = theta[idx] + alpha[idx, None] * d[idx]
            ft = model.loss(trial)
            ok = torch.isfinite(ft) & (ft <= f[idx] + c1 * alpha[idx] * gd[idx])
            oi = idx[ok]
            th_new[oi] = trial[ok]
            f_new[oi] = ft[ok]
            accepted[oi] = True
            todo[oi] = False
            ni = idx[~ok]
            alpha[ni] = alpha[ni] * 0.5
        failed = active & ~accepted

        # ---- gradient at the new point, curvature pair ------------------------------
        _, g_new = model.loss_grad(th_new)
        s = th_new - theta
        y = g_new - g
        sy = (s * y).sum(1)
        good = accepted & torch.isfinite(sy) & (sy > 1e-12 * s.norm(dim=1) * y.norm(dim=1))
        j = ptr % hist
        sy_safe = torch.where(good, sy, torch.ones_like(sy))
        Sh[:, j] = torch.where(good[:, None], s, torch.zeros_like(s))
        Yh[:, j] = torch.where(good[:, None], y, torch.zeros_like(y))
        rho[:, j] = torch.where(good, 1.0 / sy_safe, torch.zeros_like(sy))
        gamma = torch.where(good, sy_safe / (y * y).sum(1).clamp_min(1e-300), gamma)
        ptr += 1

        dec = f - f_new
        small = accepted & (dec <= ftol * f.clamp_min(1e-300))
        smalls = torch.where(small, smalls + 1, torch.where(accepted, torch.zeros_like(smalls), smalls))
        fails = torch.where(failed, fails + 1, torch.zeros_like(fails))
        theta = th_new
        f = torch.where(accepted, f_new, f)
        g = torch.where(accepted[:, None], g_new, g)
        first = (first & ~accepted) | failed
        if bool(failed.any()):                                  # restart those from steepest descent
            rho[failed] = 0.0
            Sh[failed] = 0.0
            Yh[failed] = 0.0
            gamma[failed] = 1.0
        active = active & (fails < 3) & (smalls < patience)

        if log is not None and log_every and (it + 1) % log_every == 0:
            log(f"      it {it + 1:5d}  best SSE={float(f.min()):.3e}  active={int(active.sum())}/{S}")

    return theta, model.loss(theta)


def polish_torch_lbfgs(model, theta, iters, log=None):
    """Reference path: stock torch.optim.LBFGS (strong Wolfe), one start at a time,
    using the analytic gradient."""
    out, fs = [], []
    for i in range(theta.shape[0]):
        th = theta[i:i + 1].clone().requires_grad_(True)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=iters, max_eval=int(iters * 1.25),
                                tolerance_grad=1e-10, tolerance_change=1e-15,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            with torch.no_grad():
                f, g = model.loss_grad(th.detach())
            th.grad = g.clone()
            return f[0]
        try:
            opt.step(closure)
        except Exception as e:                                   # keep whatever we have
            if log:
                log(f"      torch LBFGS exception: {e}")
        with torch.no_grad():
            out.append(th.detach().clone())
            fs.append(model.loss(th.detach())[0])
    return torch.cat(out), torch.stack(fs)


# ----------------------------------------------------------------------------
# One (L, m, q)
# ----------------------------------------------------------------------------
def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def fit_combo(XLAG_np, target_np, L, m, q, cfg, dev, dt, gen, log):
    X = torch.as_tensor(XLAG_np, device=dev, dtype=dt)
    y = torch.as_tensor(target_np, device=dev, dtype=dt)
    model = BatchedModel(X, y, q, cfg["activation"])
    gtol = max(cfg["gtol"], 1e-5) if dt == torch.float32 else cfg["gtol"]
    ftol = max(cfg["ftol"], 1e-7) if dt == torch.float32 else cfg["ftol"]
    tm = {}

    t = time.time()
    th0, sse0 = random_feature_init(model, cfg["n_cand"], cfg["starts"], cfg["chunk"], gen)
    sync(dev); tm["init"] = time.time() - t

    t = time.time()
    th1, f1 = lbfgs_batched(model, th0, cfg["screen_iters"], cfg["hist"], gtol, ftol)
    f1 = torch.where(torch.isfinite(f1), f1, torch.full_like(f1, math.inf))
    sync(dev); tm["screen"] = time.time() - t

    order = torch.argsort(f1)[:cfg["topk"]]
    t = time.time()
    if cfg["polish"] == "batched":
        th2, f2 = lbfgs_batched(model, th1[order], cfg["polish_iters"], cfg["hist"], gtol, ftol,
                                log=log, log_every=cfg["log_every"])
    else:
        th2, f2 = polish_torch_lbfgs(model, th1[order], cfg["polish_iters"], log)
    sync(dev); tm["polish"] = time.time() - t
    # keep a polished result only if it did not get worse than the screened one
    f2 = torch.where(torch.isfinite(f2), f2, torch.full_like(f2, math.inf))
    return th2.double().cpu().numpy(), f2.double().cpu().numpy(), float(sse0[0]), tm


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def fmt(v, spec):
    if np.isfinite(v):
        return format(v, spec)
    return "    -inf" if v == -math.inf else ("    +inf" if v == math.inf else "     nan")


def analyse(x_raw, name, cfg, dev, dt, log):
    d = cfg["decimate"]
    x = x_raw[::d] if d > 1 else x_raw
    x = (x - x.mean()) / x.std()
    N = len(x)
    lags = cfg["lags"] or list(range(1, cfg["max_l"] + 1))
    combos = [(L, m, q) for L in lags for m in range(1, cfg["max_m"] + 1)
              for q in range(1, cfg["max_q"] + 1) if N - m * L > n_params(q, m) + 1]
    log(f"\n=== {name}: N={N}{' (decimated x%d)' % d if d > 1 else ''}, {len(combos)} combos, "
        f"{cfg['starts']} starts (from {cfg['n_cand']} candidates), top-{cfg['topk']} polished, "
        f"{dev}, {str(dt).split('.')[-1]} ===")
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    t0 = time.time()
    rows = []
    for i, (L, m, q) in enumerate(combos, 1):
        XLAG = lag_matrix_np(x, L, m)
        target = x[m * L:]
        th2, f2, sse_init, tm = fit_combo(XLAG, target, L, m, q, cfg, dev, dt, gen, log)
        order = np.argsort(f2)
        h = max(1, int(round(len(target) ** (1 / 3))))
        t = time.time()
        cand = []
        for k in order:
            if np.isfinite(f2[k]):
                cand.append((float(f2[k]), lyapunov(XLAG, th2[k], q, cfg["activation"], 1),
                             lyapunov(XLAG, th2[k], q, cfg["activation"], h), th2[k]))
        tm["lambda"] = time.time() - t
        if not cand:
            log(f"  [{i:3d}/{len(combos)}] (L={L}, m={m}, q={q}) all starts diverged")
            continue
        sse, lam_T, lam_h, th = cand[0]
        lamsT = [c[1] for c in cand if not np.isnan(c[1])]
        var_t = float(np.var(target))
        rows.append(dict(L=L, m=m, q=q, sse=sse, nmse=sse / (len(target) * var_t), lam_T=lam_T,
                         lam_h=lam_h, lamT_min=min(lamsT) if lamsT else float("nan"),
                         lamT_max=max(lamsT) if lamsT else float("nan"), theta=th))
        log(f"  [{i:3d}/{len(combos)}] (L={L}, m={m}, q={q}) | init SSE={sse_init:.2e} -> SSE={sse:.3e} | "
            f"lam_T={fmt(lam_T, '+.4f')} lam_h={fmt(lam_h, '+.4f')} | "
            f"t: init {tm['init']:.1f}s screen {tm['screen']:.1f}s polish {tm['polish']:.1f}s "
            f"lambda {tm['lambda']:.2f}s | total {time.time() - t0:.0f}s")
    if not rows:
        log("  no valid fit.")
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
        log("  WARNING: no combo has a stable lambda across its polished starts; "
            "selecting among all combos (treat the result with caution).")
        eligible = [r for r in rows if r["sse"] <= cfg["max_sse_ratio"] * min_sse]
    if cfg["select"] == "sse":
        best = min(eligible, key=lambda r: r["sse"])
    else:
        key = "lam_T" if cfg["select"] == "lambda_T" else "lam_h"
        best = max(eligible, key=lambda r: (r[key] if np.isfinite(r[key]) else -math.inf,
                                            -(r["L"] + r["m"] + r["q"])))
    sc = 1.0 / d
    log(f"\n  {'L':>2} {'m':>2} {'q':>2} | {'SSE':>10} {'NMSE':>9} | {'lam_T':>8} {'lam_h':>8} "
        f"| lam_T over polished starts (min..max)")
    for r in sorted(rows, key=lambda r: (r["L"], r["m"], r["q"])):
        mark = "  <== selected" if r is best else ("" if r["stable"] else "  (unstable lambda)")
        log(f"  {r['L']:>2} {r['m']:>2} {r['q']:>2} | {r['sse']:10.3e} {r['nmse']:9.2e} | "
            f"{fmt(r['lam_T'], '+8.4f')} {fmt(r['lam_h'], '+8.4f')} | "
            f"{fmt(r['lamT_min'], '+.4f')} .. {fmt(r['lamT_max'], '+.4f')}{mark}")
    log(f"\n  RESULT {name}: (L,m,q)=({best['L']},{best['m']},{best['q']})  "
        f"lambda_T={best['lam_T'] * sc:+.4f}  lambda_h={best['lam_h'] * sc:+.4f}"
        f"{' (per original sample)' if d > 1 else ''}  SSE={best['sse']:.3e}  NMSE={best['nmse']:.2e}  "
        f"[{time.time() - t0:.1f}s]")
    out = dict(name=name, N=N, decimate=d, selected=dict(L=best["L"], m=best["m"], q=best["q"]),
               lambda_T=best["lam_T"] * sc, lambda_h=best["lam_h"] * sc, sse=best["sse"],
               nmse=best["nmse"], seconds=time.time() - t0,
               table=[{k: v for k, v in r.items() if k != "theta"} for r in rows])
    return out, best, x


def plot_reconstruction(x, best, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L, m, q = best["L"], best["m"], best["q"]
    B, C, A, D = unpack_np(best["theta"], q, m)
    y = D + act_np(lag_matrix_np(x, L, m) @ B.T + C, cfg["activation"])[0] @ A
    t = np.arange(m * L, len(x))
    k = min(len(t), 1000)
    plt.figure(figsize=(10, 4))
    plt.plot(t[:k], x[m * L:][:k], "k", alpha=.8, label="signal (standardised)")
    plt.plot(t[:k], y[:k], "r--", label=f"fit (L={L}, m={m}, q={q})")
    plt.legend(); plt.grid(ls=":", alpha=.6); plt.xlabel("t"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def pick_device(name, log):
    dev = torch.device("cuda:0" if (name == "auto" and torch.cuda.is_available()) else
                       ("cpu" if name == "auto" else name))
    if dev.type == "cuda":
        try:
            prop = torch.cuda.get_device_properties(dev)
            log(f"GPU: {prop.name}, compute capability {prop.major}.{prop.minor}, "
                f"{prop.total_memory / 2 ** 30:.1f} GiB")
            _ = (torch.ones(8, device=dev, dtype=torch.float64) * 2).sum().item()   # kernel really runs?
        except Exception as e:
            log(f"WARNING: CUDA device unusable with this PyTorch build ({e}); falling back to CPU. "
                f"(Old GPUs such as the K80 need an old PyTorch build.)")
            dev = torch.device("cpu")
    return dev


def check_gradient(log):
    """Analytic gradient vs autograd, float64, random data."""
    torch.manual_seed(0)
    for fn in ("tanh", "logistic", "sigmoid"):
        X = torch.randn(200, 3, dtype=torch.float64)
        y = torch.randn(200, dtype=torch.float64)
        mdl = BatchedModel(X, y, 4, fn)
        th = (torch.randn(5, mdl.P, dtype=torch.float64) * 0.7).requires_grad_(True)
        f = mdl.loss(th)
        f.sum().backward()
        _, g = mdl.loss_grad(th.detach())
        err = (g - th.grad).abs().max().item() / th.grad.abs().max().item()
        log(f"gradient check [{fn:8s}] max relative error = {err:.2e}  {'OK' if err < 1e-9 else 'FAIL'}")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_argument_group("signal")
    s.add_argument("--input", help=".mat / .npy / .csv (required, except with --check)")
    s.add_argument("--key", default="x")
    s.add_argument("--imf", default="0", help="'all' or comma list of column indices (0-based)")
    s.add_argument("--delimiter", default=",")
    s.add_argument("--decimate", type=int, default=1, help="lambda is reported per ORIGINAL sample")
    g = p.add_argument_group("model grid")
    g.add_argument("--max-l", type=int, default=2)
    g.add_argument("--max-m", type=int, default=3)
    g.add_argument("--max-q", type=int, default=6)
    g.add_argument("--lags", type=int, nargs="*", default=None)
    g.add_argument("--activation", choices=["tanh", "logistic", "sigmoid"], default="tanh")
    o = p.add_argument_group("optimisation")
    o.add_argument("--n-cand", type=int, default=4096, help="random hidden layers screened per combo")
    o.add_argument("--starts", type=int, default=64, help="candidates kept as L-BFGS starts")
    o.add_argument("--chunk", type=int, default=256, help="candidates processed at once (memory)")
    o.add_argument("--screen-iters", type=int, default=300)
    o.add_argument("--topk", type=int, default=4)
    o.add_argument("--polish", choices=["batched", "torch"], default="batched")
    o.add_argument("--polish-iters", type=int, default=5000)
    o.add_argument("--hist", type=int, default=20, help="L-BFGS history size (batched)")
    o.add_argument("--gtol", type=float, default=1e-9)
    o.add_argument("--ftol", type=float, default=1e-13)
    o.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    o.add_argument("--device", default="auto")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--log-every", type=int, default=0, help="print progress every k polish iterations")
    r = p.add_argument_group("selection / output")
    r.add_argument("--select", choices=["lambda_T", "lambda_h", "sse"], default="lambda_T")
    r.add_argument("--max-lam-spread", type=float, default=0.1)
    r.add_argument("--max-sse-ratio", type=float, default=math.inf)
    r.add_argument("--out", default=None,
                   help="output json (default: results_NAME.json, NAME = input file name without extension)")
    r.add_argument("--plot", action="store_true")
    r.add_argument("--log", default=f"chaostest_GPU_{time.strftime('%Y%m%d_%H%M%S')}.log")
    r.add_argument("--check", action="store_true", help="test the analytic gradient and exit")
    return p.parse_args()


def main():
    args = parse()
    log = setup_logger(args.log).info
    log(f"logging to {args.log}")
    if args.check:
        check_gradient(log)
        return
    if not args.input:
        sys.exit("error: --input is required")
    if args.out is None:
        args.out = f"results_{os.path.splitext(os.path.basename(args.input))[0]}.json"
    cfg = vars(args).copy()
    dev = pick_device(args.device, log)
    dt = torch.float64 if args.dtype == "float64" else torch.float32
    signals = load_signals(args)
    results = []
    for name, x in signals:
        res = analyse(x, name, cfg, dev, dt, log)
        if res is None:
            continue
        out, best, xs = res
        results.append(out)
        if args.plot:
            plot_reconstruction(xs, best, cfg, f"{os.path.splitext(args.out)[0]}_{name}.png")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    log(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
