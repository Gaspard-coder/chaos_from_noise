#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jacobian_sweep.py  --  calibrate (L, m, q) of the Jacobian / neural-network Lyapunov
estimator on systems whose largest Lyapunov exponent is KNOWN (Lorenz, Rossler,
logistic map, Henon map).

It reuses the fitting code of main_jacobian_v3.py (multi-start Levenberg-Marquardt,
analytic Jacobian, multiprocessing), so that file must sit in the same folder.
For every system and every (L, m, q) of the grid it fits the network, computes
lambda with two estimators, and compares it with the theoretical value.

INPUT: a JSON config (relative file paths are looked up from the cwd, then from the
config's folder). Minimal example:

  {"systems": [
     {"name": "lorenz",   "file": "lorenz.mat",   "key": "x", "column": 0,
      "lambda": 0.9056, "dt": 0.05},
     {"name": "logistic", "file": "logistic.mat", "key": "x",
      "lambda": 0.6931, "dt": 1}
  ]}

  name      label used in the outputs
  file,key  .mat file and variable name (key optional if the file holds one variable)
  column    which column to use if the variable is N x k (default 0 = x component)
  lambda    theoretical largest Lyapunov exponent PER TIME UNIT
            (per iteration for maps, i.e. dt = 1)
  dt        sampling interval of the data (default 1)
  decimate  keep every d-th sample (default 1)
  lags/ms/qs  optional per-system grid overriding the command-line grid

UNITS. Every lambda printed or saved is per TIME UNIT:
    lambda = lambda_per_used_sample / (dt * decimate),
    lambda_per_used_sample = lambda_per_map_step / L
because the companion map X_t -> X_{t+L} advances L samples per step. This division
by L is NOT done by main_jacobian_v3.py, which reports the raw per-step value.

TWO ESTIMATORS (identical when L = 1)
  orb : orbit-consistent. Jacobians are chained along the true orbit of the map,
        i.e. rows t, t+L, t+2L, ... ; the L interleaved chains are averaged.
  leg : legacy, as in BenSaida / main_jacobian_v3 : consecutive rows t, t+1, ...
        multiplied with the same companion structure. For L > 1 this is not an
        orbit of the map.

OUTPUT (in --outdir): sweep_<name>.json (all rows, written as soon as a system is
done; --resume skips systems already there), sweep_all.csv, sweep_summary.json,
plus tables on stdout: per system, and a cross-system ranking (mean relative error
over the systems) which is the number to look at.

Examples
    python jacobian_sweep.py --config systems.json --dry-run        # no fitting
    python jacobian_sweep.py --config systems.json --workers 32
    python jacobian_sweep.py --config systems.json --only lorenz --lags 1 2 4 8
"""

# --- keep every worker single-threaded: must be set BEFORE importing numpy ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.io

import main_jacobian_v3 as mj          # fitting code + legacy lambda estimator

ESTIMATORS = (("orb", "orbit-consistent"), ("leg", "legacy (consecutive Jacobians)"))


# ----------------------------------------------------------------------------
# Config and data
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    systems = cfg["systems"] if isinstance(cfg, dict) else cfg
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for s in systems:
        s = dict(s)
        for k in ("name", "file", "lambda"):
            if k not in s:
                raise ValueError(f"system entry {s} is missing '{k}'")
        s.setdefault("key", None)
        s.setdefault("column", 0)
        s.setdefault("dt", 1.0)
        s.setdefault("decimate", 1)
        if not os.path.exists(s["file"]) and os.path.exists(os.path.join(base, s["file"])):
            s["file"] = os.path.join(base, s["file"])
        out.append(s)
    return out


def load_signal(s):
    """1-D float64 array from a .mat file (v5/v7 via scipy, v7.3 via h5py)."""
    path, key = s["file"], s["key"]
    try:
        mat = scipy.io.loadmat(path)
        names = [k for k in mat if not k.startswith("__")]
        if key is None and len(names) == 1:
            key = names[0]
        if key not in mat:
            raise KeyError(f"{path}: variable '{key}' not found, available: {names}")
        data = np.asarray(mat[key], dtype=np.float64)
    except NotImplementedError:                      # MATLAB v7.3 = HDF5
        import h5py
        with h5py.File(path, "r") as f:
            names = list(f.keys())
            if key is None and len(names) == 1:
                key = names[0]
            if key not in f:
                raise KeyError(f"{path}: variable '{key}' not found, available: {names}")
            data = np.asarray(f[key], dtype=np.float64).T
    data = np.squeeze(data)
    if data.ndim == 2:
        if data.shape[0] < data.shape[1]:
            data = data.T
        data = data[:, int(s["column"])]
    if data.ndim != 1:
        raise ValueError(f"{path}: cannot reduce '{key}' to a 1-D series (shape {data.shape})")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: '{key}' contains NaN/Inf")
    return data.copy()


def make_grid(N, lags, ms, qs):
    combos = [(L, m, q) for L in lags for m in ms for q in qs
              if N - m * L > mj.n_params(q, m) + 1]
    combos.sort(key=lambda c: -mj.n_params(c[2], c[1]))          # heaviest first
    return combos


# ----------------------------------------------------------------------------
# Worker side (screening task is main_jacobian_v3._screen_task)
# ----------------------------------------------------------------------------
def _polish_task(task):
    """Long LM run from a screened start, then both lambda estimators (per map step)."""
    L, m, q, rank, th0 = task
    cfg = mj._G["cfg"]
    prob = mj._problem(L, m, q)
    th, sse = mj.run_lm(prob, th0, cfg["polish_nfev"], cfg["polish_tol"])
    lam_leg = lam_orb = float("nan")
    if np.isfinite(sse):
        fn = cfg["activation"]
        lam_leg = mj.lyapunov(prob.X, th, q, fn, 1)
        if L == 1:
            lam_orb = lam_leg
        else:   # chain r = rows r, r+L, r+2L, ... (lyapunov() with stride L takes rows L-1, 2L-1, ...)
            lam_orb = float(np.mean([mj.lyapunov(prob.X[r:], th, q, fn, L) for r in range(L)]))
    return L, m, q, rank, sse, lam_leg, lam_orb


def run_fits(x, combos, fit_cfg, args, label):
    t0 = time.time()
    n_starts = args.starts
    with ProcessPoolExecutor(max_workers=args.workers, initializer=mj._init_worker,
                             initargs=(x, fit_cfg)) as pool:
        # ---- pass 1: screening ------------------------------------------------
        tasks = [(L, m, q, st) for (L, m, q) in combos for st in range(n_starts)]
        screened = {c: [] for c in combos}
        done = {c: 0 for c in combos}
        n_done = 0
        for fut in as_completed([pool.submit(mj._screen_task, t) for t in tasks]):
            L, m, q, _, sse, th = fut.result()
            c = (L, m, q)
            screened[c].append((sse, th))
            done[c] += 1
            if done[c] == n_starts:
                n_done += 1
                if n_done % max(1, len(combos) // 20) == 0 or n_done == len(combos):
                    print(f"  [{label}] screen {n_done:4d}/{len(combos)} combos  "
                          f"[{time.time() - t0:7.1f}s]", flush=True)
        # ---- pass 2: polish the top-k starts of every combo --------------------
        ptasks = []
        for c, lst in screened.items():
            lst = sorted((v for v in lst if np.isfinite(v[0])), key=lambda v: v[0])
            for rank, (_, th) in enumerate(lst[:args.topk]):
                ptasks.append((*c, rank, th))
        ptasks.sort(key=lambda t: -mj.n_params(t[2], t[1]))
        polished = {c: [] for c in combos}
        for i, fut in enumerate(as_completed([pool.submit(_polish_task, t) for t in ptasks]), 1):
            L, m, q, _, sse, lam_leg, lam_orb = fut.result()
            polished[(L, m, q)].append((sse, lam_leg, lam_orb))
            if i % max(1, len(ptasks) // 10) == 0 or i == len(ptasks):
                print(f"  [{label}] polish {i:4d}/{len(ptasks)}  [{time.time() - t0:7.1f}s]", flush=True)
    return polished


# ----------------------------------------------------------------------------
# Rows, errors, selection rules
# ----------------------------------------------------------------------------
def build_rows(polished, x, N, s, tau):
    lam_th = float(s["lambda"])
    den = max(abs(lam_th), 1e-12)
    var = float(np.var(x))
    rows = []
    for (L, m, q), lst in polished.items():
        lst = sorted((v for v in lst if np.isfinite(v[0])), key=lambda v: v[0])
        if not lst:
            continue
        sse, lam_leg, lam_orb = lst[0]                   # lambda of the lowest-SSE start
        conv = 1.0 / (L * tau)                           # per map step -> per time unit

        def spread(i):                                   # over the polished starts (finite values)
            vals = [v[i] * conv for v in lst if np.isfinite(v[i])]
            return (max(vals) - min(vals)) if vals else float("nan")

        r = dict(system=s["name"], L=L, m=m, q=q, n_par=mj.n_params(q, m), sse=sse,
                 nmse=sse / ((N - m * L) * var), lam_th=lam_th,
                 lam_orb=lam_orb * conv, lam_leg=lam_leg * conv,
                 spr_orb=spread(2), spr_leg=spread(1))
        for e in ("orb", "leg"):
            lam = r["lam_" + e]
            r["rel_" + e] = abs(lam - lam_th) / den if np.isfinite(lam) else math.inf
        rows.append(r)
    return rows


def _k(v):
    return v if (v is not None and np.isfinite(v)) else math.inf


def rule_pick(rows, e, tau, max_spread):
    """BenSaida's rule: largest lambda among combos with a stable lambda over the polished starts
    (spread, expressed per used sample, <= max_spread)."""
    ok = [r for r in rows if np.isfinite(r["lam_" + e])
          and np.isfinite(r["spr_" + e]) and r["spr_" + e] * tau <= max_spread]
    if not ok:
        ok = [r for r in rows if np.isfinite(r["lam_" + e])]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["lam_" + e], -(r["L"] + r["m"] + r["q"])))


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def f4(v, w=9):
    if np.isfinite(v):
        return f"{v:+{w}.4f}"
    return f"{'-inf' if v == -math.inf else ('+inf' if v == math.inf else 'nan'):>{w}}"


def pct(v):
    return f"{100 * v:7.1f}%" if np.isfinite(v) else "    inf "


def print_row(r, e):
    print(f"     {r['L']:>2} {r['m']:>2} {r['q']:>2} | {r['nmse']:9.2e} | "
          f"{f4(r['lam_' + e])} {pct(r['rel_' + e])} | {f4(r['spr_' + e], 7)}")


def print_system(res, args):
    rows, tau = res["rows"], res["tau"]
    print(f"\n=== {res['name']}: theory {res['lambda_theory']:.4f} per time unit | N={res['N']}, "
          f"dt*decimate={tau:g} | {len(rows)} fitted combos ===")
    for e, label in ESTIMATORS:
        print(f"  -- {label}: closest to theory")
        print(f"     {'L':>2} {'m':>2} {'q':>2} | {'NMSE':>9} | {'lambda':>9} {'rel.err':>8} | {'spread':>7}")
        for r in sorted(rows, key=lambda r: r["rel_" + e])[:args.top]:
            print_row(r, e)
        pick = rule_pick(rows, e, tau, args.max_lam_spread)
        if pick is not None:
            print("     BenSaida rule (largest stable lambda):")
            print_row(pick, e)


def cross_system(results, e):
    tables = [{(r["L"], r["m"], r["q"]): r for r in res["rows"]} for res in results]
    if len(tables) < 2:
        return []
    common = set.intersection(*[set(t) for t in tables])
    agg = []
    for c in common:
        rels = [_k(t[c]["rel_" + e]) for t in tables]
        agg.append(dict(L=c[0], m=c[1], q=c[2], mean_rel=float(np.mean(rels)),
                        max_rel=float(np.max(rels)),
                        lam={res["name"]: t[c]["lam_" + e] for res, t in zip(results, tables)}))
    agg.sort(key=lambda a: (a["mean_rel"], a["max_rel"], a["L"] + a["m"] + a["q"]))
    return agg


def print_cross(results, e, label, args):
    agg = cross_system(results, e)
    if not agg:
        print(f"\n(cross-system ranking [{label}] skipped: needs >= 2 systems sharing at least one combo)")
        return agg
    names = [res["name"] for res in results]
    print(f"\n=== CROSS-SYSTEM ranking, {label} estimator "
          f"({len(agg)} combos common to {len(names)} systems) ===")
    print(f"  {'L':>2} {'m':>2} {'q':>2} | {'mean err':>8} {'max err':>8} | "
          + " ".join(f"{n[:9]:>9}" for n in names))
    for a in agg[:args.top]:
        print(f"  {a['L']:>2} {a['m']:>2} {a['q']:>2} | {pct(a['mean_rel'])} {pct(a['max_rel'])} | "
              + " ".join(f4(a['lam'][n]) for n in names))
    ref = tuple(args.ref)
    for i, a in enumerate(agg, 1):
        if (a["L"], a["m"], a["q"]) == ref:
            print(f"  reference {ref}: rank {i}/{len(agg)}, mean err {pct(a['mean_rel']).strip()}, "
                  f"max err {pct(a['max_rel']).strip()}")
            break
    else:
        print(f"  reference {ref} is not in the common grid.")
    print("  theory: " + ", ".join(f"{res['name']}={res['lambda_theory']:.4f}" for res in results))
    return agg


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_system(s, args, fit_cfg):
    path = os.path.join(args.outdir, f"sweep_{s['name']}.json")
    if args.resume and not args.dry_run and os.path.exists(path):
        print(f"\n=== {s['name']}: found {path}, skipping (--resume) ===")
        with open(path) as f:
            return json.load(f)
    d = int(s["decimate"])
    x = load_signal(s)[::d]
    x = (x - x.mean()) / x.std()                          # lambda is invariant to affine rescaling
    N, tau, lam_th = len(x), float(s["dt"]) * d, float(s["lambda"])
    combos = make_grid(N, s.get("lags", args.lags), s.get("ms", args.ms), s.get("qs", args.qs))
    print(f"\n=== {s['name']}: N={N}, dt*decimate={tau:g}, theory lambda={lam_th:.4f}/time "
          f"(= {lam_th * tau:.4f}/sample), {len(combos)} combos ===", flush=True)
    if abs(lam_th * tau) < 0.02:
        print(f"  WARNING: lambda per sample is only {lam_th * tau:.4f}: the series is heavily "
              f"oversampled, the one-step map is almost the identity and lambda is hard to "
              f"resolve. Consider a larger dt (or --decimate) and larger lags.")
    if args.dry_run:
        print(f"  dry run: {len(combos) * args.starts} screening LM runs "
              f"(<= {args.screen_nfev} evaluations each) + {len(combos) * args.topk} polish runs "
              f"(<= {args.polish_nfev} each); lags={sorted({c[0] for c in combos})}, "
              f"ms={sorted({c[1] for c in combos})}, qs={sorted({c[2] for c in combos})}")
        return None
    t0 = time.time()
    polished = run_fits(x, combos, fit_cfg, args, s["name"])
    rows = build_rows(polished, x, N, s, tau)
    if not rows:
        print("  no valid fit.")
        return None
    res = dict(name=s["name"], N=N, decimate=d, dt=float(s["dt"]), tau=tau,
               lambda_theory=lam_th, seconds=time.time() - t0, rows=rows)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res


def summarise(results, args):
    summary = dict(per_system={}, cross_system={})
    for res in results:
        summary["per_system"][res["name"]] = {
            e: dict(closest=sorted(res["rows"], key=lambda r: r["rel_" + e])[0],
                    bensaida_rule=rule_pick(res["rows"], e, res["tau"], args.max_lam_spread))
            for e, _ in ESTIMATORS}
    for e, label in ESTIMATORS:
        summary["cross_system"][e] = print_cross(results, e, label, args)[:100]
    all_rows = [r for res in results for r in res["rows"]]
    if all_rows:
        with open(os.path.join(args.outdir, "sweep_all.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    with open(os.path.join(args.outdir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nsaved {args.outdir}/sweep_all.csv, sweep_summary.json and sweep_<name>.json")
    print("CAUTION: the best (L,m,q) of one system on one realisation is partly selected by luck "
          "(min over many combos). Trust combos that rank well across systems, then validate the "
          "few winners on NEW realisations (other initial conditions / seeds).")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="JSON file describing the systems (see above)")
    p.add_argument("--only", default=None, help="comma list of system names to run")
    p.add_argument("--dry-run", action="store_true",
                   help="load the data, print grid sizes and warnings, fit nothing")
    g = p.add_argument_group("grid (can be overridden per system in the config)")
    g.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    g.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    g.add_argument("--qs", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    g.add_argument("--activation", choices=["tanh", "logistic", "sigmoid"], default="tanh")
    o = p.add_argument_group("optimisation (cheaper defaults than main_jacobian_v3.py)")
    o.add_argument("--starts", type=int, default=10, help="random starts per (L,m,q)")
    o.add_argument("--rf-cand", type=int, default=30)
    o.add_argument("--screen-nfev", type=int, default=300)
    o.add_argument("--screen-tol", type=float, default=1e-10)
    o.add_argument("--topk", type=int, default=2, help="starts polished per (L,m,q)")
    o.add_argument("--polish-nfev", type=int, default=2000)
    o.add_argument("--polish-tol", type=float, default=1e-12)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--workers", type=int, default=mj.available_cpus())
    r = p.add_argument_group("output")
    r.add_argument("--outdir", default="sweep_out")
    r.add_argument("--resume", action="store_true", help="skip systems whose sweep_<name>.json exists")
    r.add_argument("--top", type=int, default=8, help="rows shown per table")
    r.add_argument("--ref", type=int, nargs=3, default=[5, 6, 5], metavar=("L", "M", "Q"),
                   help="reference triplet whose cross-system rank is reported (BenSaida's ceiling)")
    r.add_argument("--max-lam-spread", type=float, default=0.1,
                   help="stability filter of the BenSaida-rule line: spread of lambda over the polished "
                        "starts, per used sample, must be <= this")
    return p.parse_args()


def main():
    args = parse()
    systems = load_config(args.config)
    if args.only:
        wanted = set(args.only.split(","))
        systems = [s for s in systems if s["name"] in wanted]
        if not systems:
            raise SystemExit(f"--only {args.only}: no matching system name")
    fit_cfg = dict(activation=args.activation, seed=args.seed, rf_cand=args.rf_cand,
                   screen_nfev=args.screen_nfev, screen_tol=args.screen_tol,
                   polish_nfev=args.polish_nfev, polish_tol=args.polish_tol)
    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for s in systems:
        res = run_system(s, args, fit_cfg)
        if res is not None:
            results.append(res)
            print_system(res, args)                      # partial results visible right away
    if args.dry_run or not results:
        return
    print("\n" + "=" * 100 + "\nFINAL SUMMARY")
    summarise(results, args)


if __name__ == "__main__":
    main()
