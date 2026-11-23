# Environment
- Hardware: Tesla K80 GPU
- Cuda 11.3
- PyTorch 1.12.1 + cu113
- Python 3.10
- numpy 1.26.4

The best approach is to use a python virtual environnement (venv, uv,...) to lock the version

### 3.1 Pipeline

1. Load the signal(s) and standardise. Build the grid of (L, m, q) (combos with too few samples for P parameters are skipped).
2. Screening pass. One task per (L, m, q, start). Each start is initialised from the best of --rf-cand random hidden layers (the output weights A and D are solved by linear least squares for each candidate; --rf-cand 0 gives a plain random start), then refined by LM limited to --screen-nfev evaluations.
3. Polish pass. For each combo the --topk best screened starts are refined by a long LM run (--polish-nfev, tolerance --polish-tol), and lam\_T and lam\_h are computed for each polished start.
4. Table, selection (section 2.4), JSON file and optional reconstruction plot.

Efficiency measures: BLAS threads are forced to 1 in every process (otherwise processes oversubscribe the cores); the lag matrix is built once per (L, m) per worker; the forward pass is cached so scipy's separate residual and Jacobian calls share one evaluation; the Jacobian is fully vectorised; tasks are submitted heaviest first (largest P) for load balance; the signal is sent to workers once through the pool initialiser.

Random starts are reproducible: each start uses numpy SeedSequence(\[seed, L, m, q, start\]).

### 3.2 Functions

| Function / class | Role |
| --- | --- |
| lag\_matrix(x, L, m) | Lag matrix, row t = (x\_{t-L}, ..., x\_{t-mL}) |
| act(u, fn) | Returns (psi(u), psi'(u)) for tanh, logistic or sigmoid |
| n\_params, unpack | Parameter count P and split of theta into B, C, A, D |
| Problem | Residual r = target - y\_fit and analytic Jacobian dr/dtheta with a cached forward pass; used by scipy |
| run\_lm(prob, th0, max\_nfev, tol) | One LM run; returns (theta, SSE), SSE = inf on failure |
| init\_theta(prob, rng, n\_cand) | Random-feature initialisation |
| lyapunov(XLAG, th, q, fn, stride) | Renormalised Jacobian product (section 2.3) |
| \_init\_worker, \_problem | Per-process globals and cache of Problem objects |
| \_screen\_task, \_polish\_task | The two kinds of pool tasks |
| logistic\_map, logistic\_reference\_lambda | Test signal and its reference exponent from a 200000-point orbit |
| load\_signals(args) | Synthetic signal or .mat / .npy / text file |
| analyse(x, name, cfg, workers) | Runs the full pipeline for one signal |
| plot\_reconstruction | Saves a PNG of the first 1000 points of signal and fit |

### 3.3 Options

| Option | Default | Meaning |
| --- | --- | --- |
| --signal | file | file, logistic or white |
| --input, --key, --imf, --delimiter | - , x, 0, ',' | File name; variable name in a .mat; IMF column(s): 'all' or a comma list (0-based) |
| --n, --r | 10000, 3.9 | Length of synthetic signals; logistic parameter |
| --decimate | 1 | Keep every d-th sample; lambda reported per original sample |
| --max-l, --max-m, --max-q | 2, 3, 6 | Grid L = 1..max-l, m = 1..max-m, q = 1..max-q |
| --lags | none | Explicit list of L values (overrides --max-l), e.g. --lags 1 2 4 8 |
| --activation | tanh | tanh, logistic or sigmoid |
| --starts | 20 | Random starts per (L, m, q) |
| --rf-cand | 30 | Random-feature candidates per start (0 = plain random) |
| --screen-nfev, --screen-tol | 300, 1e-10 | LM evaluation cap and tolerance in the screening pass |
| --topk | 3 | Starts polished per combo |
| --polish-nfev, --polish-tol | 3000, 1e-12 | LM evaluation cap and tolerance in the polish pass |
| --seed | 0 | Base seed |
| --workers | CPUs available to the process | Number of worker processes |
| --select | lambda\_T | lambda\_T, lambda\_h or sse |
| --max-lam-spread | 0.1 | Stability threshold (inf disables) |
| --max-sse-ratio | inf | SSE eligibility ratio |
| --out | results\_jacobian.json | Results file (rewritten after each signal) |
| --plot | off | Save one PNG per signal, named \<out>\_\<signal>.png |

Examples :
```python
    python main\_jacobian.py --signal file --input imfs.mat --key imf --imf all --workers 32 --lags 1 2 4 --max-m 3 --max-q 6
    python main_jacobian.py --signal logistic --r 3.9 --n 2000 --max-l 1 --max-m 2 --max-q 3
    python main_jacobian.py --signal white --n 10000
    python main_jacobian.py --signal file --input imfs.mat --key imf --imf all --workers 32
```
