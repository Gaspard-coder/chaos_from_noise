# Environment

- Hardware: 128 threads lame 26 GPU
- Python 3.10
- numpy 1.26.4
- scipy

The best approach is to use a python virtual environnement (venv, uv,...) to lock the version

# How it works

## Pipeline

1. Load the signal(s) and standardise. Build the grid of (L, m, q) (combos with too few samples for P parameters are skipped).
2. Screening pass. One task per (L, m, q, start). Each start is initialised from the best of --rf-cand random hidden layers (the output weights A and D are solved by linear least squares for each candidate; --rf-cand 0 gives a plain random start), then refined by LM limited to --screen-nfev evaluations.
3. Polish pass. For each combo the --topk best screened starts are refined by a long LM run (--polish-nfev, tolerance --polish-tol), and lam\_T and lam\_h are computed for each polished start.
4. Table, selection (section 2.4), JSON file and optional reconstruction plot.

Efficiency measures: BLAS threads are forced to 1 in every process (otherwise processes oversubscribe the cores); the lag matrix is built once per (L, m) per worker; the forward pass is cached so scipy's separate residual and Jacobian calls share one evaluation; the Jacobian is fully vectorised; tasks are submitted heaviest first (largest P) for load balance; the signal is sent to workers once through the pool initialiser.

Random starts are reproducible: each start uses numpy SeedSequence(\[seed, L, m, q, start\]).

## Functions

| Function / class                           | Role                                                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| lag\_matrix(x, L, m)                       | Lag matrix, row t = (x\_{t-L}, ..., x\_{t-mL})                                                         |
| act(u, fn)                                 | Returns (psi(u), psi'(u)) for tanh, logistic or sigmoid                                                |
| n\_params, unpack                          | Parameter count P and split of theta into B, C, A, D                                                   |
| Problem                                    | Residual r = target - y\_fit and analytic Jacobian dr/dtheta with a cached forward pass; used by scipy |
| run\_lm(prob, th0, max\_nfev, tol)         | One LM run; returns (theta, SSE), SSE = inf on failure                                                 |
| init\_theta(prob, rng, n\_cand)            | Random-feature initialisation                                                                          |
| lyapunov(XLAG, th, q, fn, stride)          | Renormalised Jacobian product (section 2.3)                                                            |
| \_init\_worker, \_problem                  | Per-process globals and cache of Problem objects                                                       |
| \_screen\_task, \_polish\_task             | The two kinds of pool tasks                                                                            |
| logistic\_map, logistic\_reference\_lambda | Test signal and its reference exponent from a 200000-point orbit                                       |
| load\_signals(args)                        | Synthetic signal or .mat / .npy / text file                                                            |
| analyse(x, name, cfg, workers)             | Runs the full pipeline for one signal                                                                  |
| plot\_reconstruction                       | Saves a PNG of the first 1000 points of signal and fit                                                 |

## Options

| Option                             | Default                       | Meaning                                                                            |
| ---------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------- |
| --signal                           | file                          | file, logistic or white                                                            |
| --input, --key, --imf, --delimiter | - , x, 0, ','                 | File name; variable name in a .mat; IMF column(s): 'all' or a comma list (0-based) |
| --n, --r                           | 10000, 3.9                    | Length of synthetic signals; logistic parameter                                    |
| --decimate                         | 1                             | Keep every d-th sample; lambda reported per original sample                        |
| --max-l, --max-m, --max-q          | 2, 3, 6                       | Grid L = 1..max-l, m = 1..max-m, q = 1..max-q                                      |
| --lags                             | none                          | Explicit list of L values (overrides --max-l), e.g. --lags 1 2 4 8                 |
| --activation                       | tanh                          | tanh, logistic or sigmoid                                                          |
| --starts                           | 20                            | Random starts per (L, m, q)                                                        |
| --rf-cand                          | 30                            | Random-feature candidates per start (0 = plain random)                             |
| --screen-nfev, --screen-tol        | 300, 1e-10                    | LM evaluation cap and tolerance in the screening pass                              |
| --topk                             | 3                             | Starts polished per combo                                                          |
| --polish-nfev, --polish-tol        | 3000, 1e-12                   | LM evaluation cap and tolerance in the polish pass                                 |
| --seed                             | 0                             | Base seed                                                                          |
| --workers                          | CPUs available to the process | Number of worker processes                                                         |
| --select                           | lambda\_T                     | lambda\_T, lambda\_h or sse                                                        |
| --max-lam-spread                   | 0.1                           | Stability threshold (inf disables)                                                 |
| --max-sse-ratio                    | inf                           | SSE eligibility ratio                                                              |
| --out                              | results\_jacobian.json        | Results file (rewritten after each signal)                                         |
| --plot                             | off                           | Save one PNG per signal, named \<out>\_\<signal>.png                               |

Examples :

```bash
    python main\_jacobian.py --signal file --input imfs.mat --key imf --imf all --workers 32 --lags 1 2 4 --max-m 3 --max-q 6
    python main_jacobian.py --signal logistic --r 3.9 --n 2000 --max-l 1 --max-m 2 --max-q 3
    python main_jacobian.py --signal white --n 10000
    python main_jacobian.py --signal file --input imfs.mat --key imf --imf all --workers 32
```

# Improving the function

After a first batch on references with (5,6,5), the method seems promising. We will try now to use these references to fine tune the parameters and achieve a better approximation.

To achieve that, we created `jacobian_sweep.py`. Starting from the references systems specified in `systems_example.json`, it tries to find empirically the best set of parameters to approxmate the Lyapunov Exponent.

NB: it reuses the main_jacobian_v3.py function so it needs it in the same directory.

Example of use:

```bash
   python jacobian_sweep.py --config systems.json --lags 1 2 --ms 1 2 3 4 --qs 1 2 3 4 5 6 7 8 9 10
   python main_jacobian_v4.py --input deter_test_sweep --pattern "DET_*.mat" --key x --auto-dt --lags 2 --ms 4 --qs 10 --out det_results.json
```

## Details

It imports your main_jacobian_v3.py (same folder) and reuses its multi-start LM fitting. For each system and each (L, m, q) in the grid, it compares λ̂ to your theoretical value. Defaults are L 1–5, m 1–6, q 1–5, which includes (5, 6, 5). You can override the grid per system in the JSON.

It prints a per-system table and a cross-system ranking by mean relative error, with the rank of (5, 6, 5) reported. Results are saved per system, and --resume skips finished systems.

Things to know:

L division. For L > 1 the companion map advances L samples per step, so λ must be divided by L to be per sample. main_jacobian_v3.py doesn't do this, and it multiplies consecutive Jacobians, which is only a true orbit when L = 1. I compute both a legacy and an orbit-consistent estimator, both divided by L; they coincide when L = 1. Please check my reading of your lyapunov(), because it affects every L > 1 result you've seen.
Sampling interval. Set dt to the value you used to generate the flows. If λ·dt is below about 0.02, the script warns that the series is oversampled. The dt values in the template (0.05 for Lorenz, 0.2 for Rössler) are my guesses. The theoretical λ values (0.9056, 0.0714, ln 2, 0.4192) hold only for the classical parameters named in the template.
Maps versus flows. The logistic and Hénon maps will favour small m, since their true embedding is 1 and 2. Lorenz and Rössler are closer to your oscillatory IMFs, so weight them more.
Selection by luck. The minimum error over about 150 combos is partly luck. Trust combos that rank well across all four systems, then validate the few winners on new realizations.

## Orbit-Consistent (orb) vs. Legacy (leg) Estimator

When the neural network fits a delay-embedded map with lag L, it learns the L-step transition:
Xt+L​=F(Xt​),J(Xt​)=∂Xt​∂Xt+L​​

The two estimators differ in how they multiply these Jacobians to compute the Lyapunov exponent:

    Orbit-consistent (orb): Steps along the actual orbit of the L-step map—multiplying J(Xt+kL​)⋯J(Xt+L​)J(Xt​) (stepping by L, and averaging over the L interleaved sub-orbits). By the chain rule, each matrix maps the tangent space at Xt+kL​ directly into the tangent space at Xt+(k+1)L​, giving the true derivative of FK.

    Legacy (leg): Multiplies Jacobians at every consecutive index—J(Xt+K−1​)⋯J(Xt+1​)J(Xt​) (stepping by 1 instead of L).

        When L=1, both estimators are mathematically identical (which is why (1, 3, 10) and (1, 4, 8) have the exact same numbers in both tables).

        When L>1, leg is mathematically inconsistent: J(Xt​) maps a perturbation from t→t+L, not t→t+1, so it composes Jacobians across mismatched tangent spaces. That is why no L=2 or L=3 combo makes the top 8 in leg, whereas L=2 performs very well in orb.

Bottom line: Always use orb whenever L>1.

## Applying the orbit-consistent approach

As this approach gives better results, and in order to use the same hyperparameters, we created `main_jacobian_v4.py` to try the new best set of parameters (2,4,10).

## Dynamically choosing the parameters + verification gates

Example:

```bash
python dynamical_jacobian.py \
  --input imf_folder \
  --pattern "IMF_*.mat" \
  --key x \
  --dt 1.0 \
  --period-method psd \
  --lag-method ami \
  --dim-method cao \
  --m-min 2 --m-max 6 \
  --qs 2 3 4 5 6 8 10 \
  --max-nmse 0.10 \
  --max-le-step 0.50 \
  --n-surrogates 19 \
  --parlitz-check \
  --out imf_verified_results
```

### Linear Degeneracy

Look at the two diagnostic numbers in your output: NMSE = 4.17e-17 and spread = 0.1155. Both tell the exact story of why a pure line gives +0.16067 instead of 0.1. Collinear Degeneracy (Over-parameterized Network on a 1D Line)A linear ramp $x_t = \alpha t + \beta$ lies strictly on a 1-dimensional straight line inside the 4-dimensional delay space $X_t = [x_{t+6}, x_{t+4}, x_{t+2}, x_t]$:$$x_{t+2k} = x_t + k \Delta$$Because all 10,000 data points sit on a single 1D line in $\mathbb{R}^4$:The data only constrains the network's output $F(X)$ and directional derivative along that 1D line (where the true eigenvalue is $1$, i.e., $\lambda = \ln 1 = 0$).In the 3 orthogonal directions perpendicular to that line, there is zero data. Your 61-parameter network (q=10, m=4) fits the 1D line to machine precision (NMSE = 4.17e-17) while leaving random, unconstrained slopes in the 3 transverse directions.When the QR decomposition multiplies the $4 \times 4$ companion Jacobians $J_t$, the largest eigenvalue of $J_t$ is no longer the true $1.0$ along the trajectory—it becomes whichever random transverse direction happens to have a spectral radius $> 1$ (here $e^{0.321} \approx 1.38$).2. Why spread = 0.1155 Already Flagged ItNotice that spread = 0.1155 is nearly 10× larger than for Lorenz (0.0122). Because the 3 transverse directions have zero effect on the training loss (NMSE ~ 1e-17), different random initializations get completely different transverse slopes and therefore completely different $\lambda$ estimates.(Note: in select_bensaida, your filter r["spread"] * tau <= max_spread with tau = 1 and max_spread = 0.1 actually rejected (2, 4, 10), and only fell back to it because there were no other combos in your 1-element grid).3. Non-Stationarity (No Attractor)Just as with MATLAB's lyapunovExponent, a linear ramp is non-stationary and unbounded—it never revisits a neighborhood in phase space. Locally, any linear map $x_{t+1} = a x_t + b$ with $a = 1$ has $\lambda = 0$, which a minimal model (L=1, m=1, q=1) recovers immediately:Bashpython main_jacobian_v4.py --input deter_test_sweep/DET_linear_r01.mat --key x --lags 1 --ms 1 --qs 1
With (L=1, m=1, q=1), there are no transverse directions to hallucinate spurious expansion, and lambda_orb will drop to ~0.0000.If you want main_jacobian_v4.py to handle low-dimensional/linear signals automatically without hallucinating transverse chaos, you can either run a grid that includes smaller (m, q) (e.g., --ms 1 2 3 4 --qs 1 2 5 10) with an AIC/BIC penalty on n_par, or add a tiny L2 weight decay (ridge) to the Levenberg–Marquardt objective so unconstrained transverse weights shrink to zero.
