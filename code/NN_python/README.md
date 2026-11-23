# Environment
- Hardware: Tesla K80 GPU
- Cuda 11.3
- PyTorch 1.12.1 + cu113
- Python 3.10
- numpy 1.26.4
- scipy

The best approach is to use a python virtual environnement (venv, uv,...) to lock the version

# How it works

## Pipeline

For every (L, m, q) of the grid, sequentially:

1. Init. --n-cand random hidden layers (B, C) are drawn (scale log-uniform between 0.3 and 5). For each, the output weights (A, D) are obtained in closed form from the normal equations (batched, with a tiny ridge, in float64). The --starts best candidates by SSE become the starting points. This replaces the previous 1000 Adam iterations on 1024 random starts.
2. Screen. Batched L-BFGS on all starts for --screen-iters iterations.
3. Polish. The --topk best starts continue for --polish-iters iterations, either with the batched L-BFGS (default) or with the stock torch.optim.LBFGS (--polish torch, strong-Wolfe line search, one start at a time, kept as a reference).
4. Lambda (CPU, numpy, float64) for each polished start, then the same table and selection as the CPU script.

The log line of each combo reports the initial SSE, the final SSE, both lambdas and the time of each stage (init, screen, polish, lambda), so the cost of every stage can be read directly.

## The batched L-BFGS (lbfgs\_batched)

The S starts are independent problems stored as a (S, P) tensor. Every start has its own L-BFGS history, but all operations are vectorised over starts so one set of kernels advances all of them.

- Gradient: analytic (no autograd graph), computed together with the loss; --check compares it with autograd.
- Direction: standard two-loop recursion with history --hist (default 20), initial scaling gamma = s'y / y'y of the latest valid pair. Circular buffers; a skipped pair is stored as zeros (rho = 0), which makes it inert.
- Safeguards: if the direction is not a descent direction (or is NaN) the start is reset to steepest descent with an empty history.
- Line search: Armijo backtracking (c1 = 1e-4, step halving, at most 25 trials), evaluated only on the starts that have not yet been accepted. The first step after a reset is min(1, 1/||g||\_1).
- Curvature pair (s, y) is stored only if s'y > 1e-12 ||s|| ||y||.
- Per-start stopping: max-norm of the gradient below --gtol, or relative decrease below --ftol for 10 consecutive accepted iterations, or 3 consecutive line-search failures (the history is reset after each failure). The loop ends when every start has stopped or at the iteration cap.
- Starts with non-finite loss or gradient are deactivated and ignored in the final ranking.

Not implemented: a Wolfe curvature condition (Armijo only), and compaction of finished starts (they still take part in the tensor operations, which is harmless for 4 to 64 starts).

## Functions

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

## Options

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
