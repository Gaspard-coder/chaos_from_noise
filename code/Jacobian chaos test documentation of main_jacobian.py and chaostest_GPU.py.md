# Jacobian chaos test: documentation of main\_jacobian.py and chaostest\_GPU.py

Version of 22 November 2026. This document describes the two Python scripts that estimate the largest Lyapunov exponent of a time series (for example each IMF of an EMD) with the neural-network Jacobian method of computing\_lyapunov\_exponent.pdf: what they compute, how the pipelines work, every option, how to read the output, and what was validated.

## 1. Overview

Both scripts fit the one-step map x\_t = f(x\_{t-L}, ..., x\_{t-mL}) + noise with a one-hidden-layer tanh network, for every triplet (L, m, q) of a grid, compute the Lyapunov exponent lambda from the Jacobian of the fitted map, and select one model. They differ only in how the network is fitted.

|  | main\_jacobian.py | chaostest\_GPU.py |
| --- | --- | --- |
| Optimiser | Levenberg-Marquardt (scipy least\_squares, method 'lm'), uses the sum-of-squares structure | L-BFGS written in PyTorch, batched: one history and one line search per start |
| Hardware | CPU only, one process per task (ProcessPoolExecutor) | GPU if available, otherwise CPU |
| Parallel axis | (L, m, q) x random start, spread over processes | random starts inside one (L, m, q), vectorised |
| Default precision | float64 | float64 (float32 optional) |
| Typical use | nodes that only run Python, no GPU needed | check whether L-BFGS matches LM, GPU nodes |

Both scripts are self-contained (no dependency on utils.py or models.py) and share the same signal loading, lambda estimators, selection rule and output format, so their results can be compared line by line.

They estimate lambda; they do not compute the test statistic or p-value of BenSaida's test.

## 2. Method

### 2.1 Model

x\_hat\_t = D + sum\_{j=1..q} A\_j \* psi( C\_j + sum\_{i=1..m} B\_ji \* x\_{t-iL} ),  psi = tanh (default), logistic or sigmoid.

The parameter vector is theta = \[B (q*m values, row-major: B\_11..B\_1m, B\_21, ...), C (q), A (q), D (1)\], so P = q*m + 2q + 1 parameters. The lag matrix has one row per t = mL .. N-1 and columns (x\_{t-L}, ..., x\_{t-mL}); T = N - mL rows.

The signal is standardised (zero mean, unit variance) before fitting. Lambda is invariant to affine rescaling of x, and conditioning improves. With --decimate d, every d-th sample is kept first and the result is then standardised.

### 2.2 Objective and fit quality

SSE = sum over t of (x\_t - x\_hat\_t)^2, minimised over theta. The scripts also print NMSE = SSE / (T \* var(target)). NMSE close to 1 means the model is no better than the mean (noise-like signal); NMSE much smaller than 1 means there is deterministic one-step structure. A poor reconstruction of a noise-dominated IMF is therefore expected, not a bug.

### 2.3 Lyapunov estimators

The derivative of the map with respect to the i-th lag is df/dx\_{t-iL} = sum\_j A\_j \* psi'(u\_j(t)) \* B\_ji, evaluated along the observed series. These derivatives form the first row of the companion Jacobian J\_t; the sub-diagonal is the identity (shift). Starting from v\_0 = e\_1, the scripts iterate v\_k = J\_{t\_k} v\_{k-1}, renormalising v each step, and accumulate the logarithm of the norm before renormalisation:

lambda = (1/M) \* sum\_k log ||J\_{t\_k} v\_{k-1}||

This equals (1/2M) ln((T\_M U0)'(T\_M U0)) of the PDF with U0 = e\_1, but cannot underflow or overflow (the previous GPU script clamped the unnormalised product at 1e-300, which silently floored very negative exponents).

Two estimators are reported:

- lam\_T: all consecutive Jacobians (stride 1, M = T). Lowest variance; recommended as the point estimate.
- lam\_h: every h-th Jacobian with h = round(T^(1/3)) (indices h-1, 2h-1, ... in 0-based form), so M is about T^(2/3). This is the sub-sampling used by the MATLAB code and suggested by the PDF.

If a Jacobian vector becomes exactly zero (saturated network, derivative below 1e-300) the exponent is reported as -inf. With --decimate d, the reported lambda is divided by d so that it is per original sample and comparable between IMFs.

### 2.4 Model selection

For every (L, m, q) the best polished start (smallest SSE) represents the combo. A combo is eligible if:

1. its SSE is at most --max-sse-ratio times the smallest SSE of the grid (default: no restriction), and
2. its lambda is stable: lam\_T varies by at most --max-lam-spread (default 0.1) across the polished starts of that combo, or all of them are below -0.5 (then only the sign matters).

Among eligible combos the one with the largest lambda is selected (BenSaida's maximum-lambda rule; --select lambda\_T or lambda\_h), ties broken by the smallest L+m+q. With --select sse the smallest SSE is used instead. If no combo is stable, a warning is printed and all combos are considered.

The stability condition exists because of an observed failure: on a noiseless one-dimensional map, an over-embedded fit (m = 2 or more) is only constrained on the orbit, so two fits with almost identical SSE can have very different derivatives off the orbit. One such fit gave lambda = +1.357 against a true value of 0.497 and would have won the maximum-lambda rule.

## 3. main\_jacobian.py (CPU, Levenberg-Marquardt)

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

Example: python main\_jacobian.py --signal file --input imfs.mat --key imf --imf all --workers 32 --lags 1 2 4 --max-m 3 --max-q 6

## 4. chaostest\_GPU.py (batched L-BFGS in PyTorch)

### 4.1 Pipeline

For every (L, m, q) of the grid, sequentially:

1. Init. --n-cand random hidden layers (B, C) are drawn (scale log-uniform between 0.3 and 5). For each, the output weights (A, D) are obtained in closed form from the normal equations (batched, with a tiny ridge, in float64). The --starts best candidates by SSE become the starting points. This replaces the previous 1000 Adam iterations on 1024 random starts.
2. Screen. Batched L-BFGS on all starts for --screen-iters iterations.
3. Polish. The --topk best starts continue for --polish-iters iterations, either with the batched L-BFGS (default) or with the stock torch.optim.LBFGS (--polish torch, strong-Wolfe line search, one start at a time, kept as a reference).
4. Lambda (CPU, numpy, float64) for each polished start, then the same table and selection as the CPU script.

The log line of each combo reports the initial SSE, the final SSE, both lambdas and the time of each stage (init, screen, polish, lambda), so the cost of every stage can be read directly.

### 4.2 The batched L-BFGS (lbfgs\_batched)

The S starts are independent problems stored as a (S, P) tensor. Every start has its own L-BFGS history, but all operations are vectorised over starts so one set of kernels advances all of them.

- Gradient: analytic (no autograd graph), computed together with the loss; --check compares it with autograd.
- Direction: standard two-loop recursion with history --hist (default 20), initial scaling gamma = s'y / y'y of the latest valid pair. Circular buffers; a skipped pair is stored as zeros (rho = 0), which makes it inert.
- Safeguards: if the direction is not a descent direction (or is NaN) the start is reset to steepest descent with an empty history.
- Line search: Armijo backtracking (c1 = 1e-4, step halving, at most 25 trials), evaluated only on the starts that have not yet been accepted. The first step after a reset is min(1, 1/||g||\_1).
- Curvature pair (s, y) is stored only if s'y > 1e-12 ||s|| ||y||.
- Per-start stopping: max-norm of the gradient below --gtol, or relative decrease below --ftol for 10 consecutive accepted iterations, or 3 consecutive line-search failures (the history is reset after each failure). The loop ends when every start has stopped or at the iteration cap.
- Starts with non-finite loss or gradient are deactivated and ignored in the final ranking.

Not implemented: a Wolfe curvature condition (Armijo only), and compaction of finished starts (they still take part in the tensor operations, which is harmless for 4 to 64 starts).

### 4.3 Functions

| Function / class | Role |
| --- | --- |
| setup\_logger | Console + file logging (file keeps timestamps) |
| lag\_matrix\_np, act\_np, unpack\_np, n\_params, lyapunov | Same as the CPU script (numpy side) |
| BatchedModel | S networks at once: unpack, act, loss(theta), loss\_grad(theta) returning (SSE, gradient) |
| random\_feature\_init | Random hidden layers + closed-form output layer, processed in chunks of --chunk candidates |
| lbfgs\_batched | The optimiser of section 4.2 |
| polish\_torch\_lbfgs | Reference polish with torch.optim.LBFGS using the analytic gradient |
| fit\_combo | Init, screen, polish for one (L, m, q), with stage timings |
| analyse | Loop over the grid, lambda, table, selection, results dictionary |
| pick\_device | Chooses cuda:0 or cpu; prints GPU name and compute capability; falls back to CPU if a CUDA kernel cannot run |
| check\_gradient | Analytic gradient versus autograd for the three activations |

### 4.4 Options

Signal, grid, selection and output options are identical to the CPU script (section 3.3), except that the default output file is results\_gpu.json and a log file chaostest\_GPU\_\<timestamp>.log is written (--log). Specific options:

| Option | Default | Meaning |
| --- | --- | --- |
| --n-cand | 4096 | Random hidden layers screened per combo |
| --starts | 64 | Candidates kept as L-BFGS starts |
| --chunk | 256 | Candidates processed at once in the init (memory control) |
| --screen-iters | 300 | L-BFGS iterations for all starts |
| --topk | 4 | Starts polished |
| --polish | batched | batched or torch |
| --polish-iters | 5000 | Iteration cap of the polish |
| --hist | 20 | L-BFGS history size (batched path) |
| --gtol, --ftol | 1e-9, 1e-13 | Stopping tolerances (raised automatically to 1e-5 and 1e-7 in float32) |
| --dtype | float64 | float64 or float32 |
| --device | auto | auto, cpu, cuda:0, ... |
| --seed | 0 | Seed of the random generator on the chosen device |
| --log-every | 0 | Print polish progress every k iterations |
| --check | off | Test the analytic gradient and exit |

Example: python chaostest\_GPU.py --signal file --input imfs.mat --key imf --imf all --lags 1 2 4 --polish-iters 5000

### 4.5 Device notes

- Memory: the large tensors have shape (S, N, q) in float64. For S = 64, N = 10000, q = 6 each is about 31 MB and a loss-and-gradient evaluation holds roughly six of them, so the optimisation stage needs a few hundred MB. The init stage needs about chunk x N x (q+1) x 8 bytes per tensor (about 120 MB per tensor for chunk = 256); lower --chunk if memory is short.
- Precision: float64 is the default because the tolerances are far below float32 resolution. On GPUs with weak float64 throughput (most consumer cards) --dtype float32 is possible, but expect SSE to stall around 1e-6 to 1e-7 relative to the signal.
- Tesla K80: the card is Kepler (compute capability 3.7), supported only by old PyTorch builds. The script prints the card's properties and, if a test kernel fails, continues on the CPU with a warning. Use nvidia-smi and a one-line torch.cuda test on gpu1 to gpu3 to see which machines have a usable card.
- Reproducibility: the generator is seeded, but GPU reductions are not guaranteed bit-identical between runs.

## 5. Inputs and outputs

### 5.1 Signals

--signal logistic (parameter --r, length --n), --signal white (standard normal noise), or --signal file with a .mat file (variable --key), a .npy file, or a text file (--delimiter). A 2-D array is interpreted as one IMF per column; if it has fewer rows than columns it is transposed automatically. --imf selects 'all' or a list such as 0,2,3.

### 5.2 Console and log

Per signal the scripts print the header (N, number of combos, starts), progress, a table with one line per (L, m, q), and a final RESULT line.

| Column | Meaning |
| --- | --- |
| SSE, NMSE | Fit quality of the best polished start (section 2.2) |
| lam\_T, lam\_h | The two estimators for that start |
| min .. max | Range of lam\_T over the polished starts of the combo; a wide range means the Jacobian is not determined by the fit |
| selected / (unstable lambda) | Selection mark; unstable combos are not eligible |

The RESULT line gives the selected triplet, lambda\_T, lambda\_h, SSE and NMSE.

### 5.3 JSON file

A list with one entry per signal: name, N (after decimation), decimate, selected {L, m, q}, lambda\_T and lambda\_h (per original sample), sse, nmse, seconds, and table (one dictionary per combo with L, m, q, sse, nmse, lam\_T, lam\_h, lamT\_min, lamT\_max, stable). Values of -inf are written as -Infinity, which Python's json module reads back correctly but strict JSON parsers reject.

## 6. Recommended workflow

1. python chaostest\_GPU.py --check on the target machine (the three gradient errors should be around 1e-16).
2. Validate on known signals: logistic r = 3.9 (reference 0.4967) and white noise (NMSE near 1, negative lambda). A small grid such as --max-l 1 --max-m 2 --max-q 3 takes seconds to a minute.
3. Run one IMF with the grid you intend to use and read the per-stage times (GPU log) or the screen and polish time stamps (CPU) before launching everything.
4. Run all IMFs (--imf all), keep the JSON, and compare lam\_T, NMSE and selected triplets between the two scripts.
5. Treat a result as reliable only if the selected combo is stable, NMSE is clearly below 1, and lam\_T is consistent between the two optimisers.

## 7. Choosing parameters

- Lags. IMFs are oscillatory and smooth, so L = 1 or 2 gives nearly collinear inputs. Use L of about a quarter of the mean period (or the first minimum of the autocorrelation), for example --lags 1 2 4 8, or decimate the IMF to roughly 6 to 10 samples per period with --decimate (this also cuts the cost linearly in N).
- Grid size. The cost of a fit grows with P = q\*m + 2q + 1 and linearly with N. Start with --max-m 3 --max-q 6 and enlarge only if the results ask for it.
- Starts and caps. The fit of a chaotic map needs many iterations: in the CPU tests on a 2000-point logistic map, LM reached SSE around 1e-4 after 100 evaluations, 1e-6 to 1e-3 after 300, 1e-7 after 1000 and 1e-9 after 3000, and the spread between starts was four orders of magnitude. Hence several starts, a short screening and a long polish of only a few of them.
- Estimator. Prefer lam\_T. In the logistic tests lam\_h (M about T^(2/3)) was off by 0.1 to 0.15 even for near-perfect fits, because it averages only a few hundred derivatives.

## 8. Troubleshooting

| Symptom | Likely cause and action |
| --- | --- |
| SSE identical for all q | Neurons are identical (symmetric start). Use random starts (the default); do not use constant initial values |
| (unstable lambda) on many combos | Over-embedding or under-fitting; lower --max-m, check NMSE, increase --topk and the polish cap |
| NMSE close to 1 for an IMF | The one-step map is noise-dominated; negative lambda is the expected answer |
| lambda = -inf | Saturated network (zero derivative); typical for noise, the sign is what matters |
| SSE far above the best seen for a combo | Too few starts or too short a polish; raise --starts, --polish-nfev or --polish-iters |
| No valid fit | Every start diverged or too few samples for P; reduce the grid |
| CUDA error or warning at start | PyTorch build does not support the card; the script falls back to CPU. Install a build matching the GPU or use another node |
| Out of memory | Lower --chunk and --starts, or use --dtype float32 |
| CPU script slow despite many cores | Check --workers and the number of cores the job is allowed to use; the thread variables are already forced to 1 |

## 9. Validation performed

Tests were run in a sandbox without GPU (CPU only, one core, PyTorch 2.15, float64), so the K80 itself was not tested.

| Test | Result |
| --- | --- |
| Logistic r = 3.9, reference lambda 0.4967 (long orbit) | CPU script, N = 2000: q = 1 cannot fit (SSE about 570); q = 2 gives SSE 2.5e-8, lam\_T 0.5030 (m = 1) and 0.5038 (m = 2); lam\_h 0.350 (m = 1) and 0.577 (m = 2) |
| Same, GPU script (L-BFGS, on CPU) | SSE 2e-7 to 9e-7; lam\_T 0.5030 and 0.5038, identical to the CPU script at 3 decimals; same combo selected |
| Same, --polish torch (stock LBFGS) | SSE 5e-5 for (1, 1, 2), against 3.6e-7 for the batched L-BFGS in the same time |
| Logistic r = 3.9, N = 10000, m = 1, q up to 3 | CPU script selected (1, 1, 2): SSE 1.9e-8, lam\_T 0.4994, lam\_h 0.5691. GPU script selected (1, 1, 3): SSE 1.0e-6, lam\_T 0.4994, lam\_h 0.5691 |
| White noise, N = 2000, CPU script | NMSE about 0.99; lam\_T negative (-1.41 for the selected combo, -inf for saturated fits) |
| File input, 2-D array with rows = IMFs, --imf all, --decimate 2 | Auto-transpose and decimation work; the logistic column gave 0.5016 per original sample |
| Analytic gradient versus autograd | Relative error about 1e-16 for tanh, logistic and sigmoid |

The timings printed during these tests are not benchmarks: the sandbox has one core and the two N = 10000 runs shared it. The file-input path of chaostest\_GPU.py is a copy of the CPU one and was not run separately.

## 10. Limitations and caveats

- Only point estimates of lambda are produced; the variance estimate and p-value of the original test are not implemented.
- The companion-matrix Jacobian is built exactly as in the original MATLAB code. For L > 1 the shift structure treats the lags as consecutive state components, and lam\_h multiplies non-consecutive Jacobians; both are kept for consistency with the original method and should be checked on a known signal before trusting results for L > 1.
- The maximum-lambda rule favours positive exponents when many combos are compared; the stability filter and the SSE and NMSE columns are there to limit this, but they do not remove it.
- SSE measures one-step prediction. A small SSE does not guarantee correct derivatives, especially for noiseless data with over-embedding (section 2.4).
- The batched L-BFGS uses an Armijo line search and ended with a higher SSE than LM in the tests (about 1e-7 against 1e-8 at N = 2000). Lambda was identical, but compare both scripts on your IMFs before relying on L-BFGS alone.
- Decimation uses plain sub-sampling without an anti-aliasing filter, which is adequate for narrow-band IMFs sampled well above their frequency.
- The scripts do not use the same random streams, so starts differ between them; agreement of the results is evidence, not an identity.
