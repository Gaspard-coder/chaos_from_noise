import logging
import sys
import time
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt

# Assuming these are available in your directory
try:
    from utils import lag_matrix_torch
    from models import activation_fn_torch
except ImportError:
    # Fallback placeholders if modules aren't immediately present in the environment
    def lag_matrix_torch(Y, lags, m):
        num_obs = len(Y)
        YLag = torch.full((num_obs, m), float('nan'), device=Y.device)
        for i in range(1, m + 1):
            L = lags * i
            YLag[L:, i-1] = Y[:-L]
        return YLag[m * lags:]

    def activation_fn_torch(u, active_fn):
        if active_fn == 'tanh':
            return torch.tanh(u)
        elif active_fn == 'logistic':
            return 1.0 / (1.0 + torch.exp(-u))
        elif active_fn == 'sigmoid':
            return u * (1.0 + torch.abs(u / 2.0)) / (2.0 + torch.abs(u) + (u**2) / 2.0)
        return torch.tanh(u)


def setup_logger(log_path):
    """
    Configures a logger that writes every message both to stdout (console)
    and to `log_path` with timestamps, flushing on every entry.
    """
    logger = logging.getLogger("chaostest_gpu")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    # File handler: includes timestamp and level, UTF-8 safe for box-drawing chars
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                                datefmt='%Y-%m-%d %H:%M:%S'))

    # Console handler: keeps clean terminal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Also route Python warnings (e.g., PyTorch/NumPy) to the same handlers
    logging.captureWarnings(True)
    py_warnings_logger = logging.getLogger("py.warnings")
    py_warnings_logger.handlers.clear()
    py_warnings_logger.addHandler(file_handler)
    py_warnings_logger.addHandler(console_handler)
    py_warnings_logger.propagate = False

    return logger


def compute_lyapunov(XLAG, B, C, A, active_fn):
    """
    Computes the Lyapunov exponent by tracking the Jacobian over time,
    exactly mirroring the MATLAB implementation in chaostest_theta.m.
    """
    # Ensure double precision to avoid underflow/overflow during long multiplications
    B_d = B.double()
    C_d = C.double()
    A_d = A.double()
    XLAG_d = XLAG.double()

    T_full, m = XLAG_d.shape
    u = XLAG_d @ B_d.T + C_d

    # Derivative of activation functions
    if active_fn == 'tanh':
        deriv_u = 1.0 - torch.tanh(u)**2
    elif active_fn == 'logistic':
        sig = 1.0 / (1.0 + torch.exp(-u))
        deriv_u = sig * (1.0 - sig)
    elif active_fn == 'sigmoid':
        abs_u = torch.abs(u)
        deriv_u = 8.0 * (1.0 + abs_u) / (3.0 + (1.0 + abs_u)**2)**2
    else:
        deriv_u = 1.0 - torch.tanh(u)**2

    # Jacobian of F(X) with respect to the lagged variables
    dy_dxlag = (deriv_u * A_d.T) @ B_d  # Shape: (T_full, m)

    # Determine the step size 'h' and blocks 'M'
    h = int(round(T_full ** (1.0 / 3.0)))
    if h < 1:
        h = 1

    # Equivalent to MATLAB's newIndex = (h:h:T), zero-indexed for PyTorch
    indices = torch.arange(h - 1, T_full, h, device=XLAG.device)
    M = len(indices)

    dy_dxlag_sub = dy_dxlag[indices]

    # Initialize U0 vector = [1; 0; ...; 0]
    vec = torch.zeros((m, 1), device=XLAG.device, dtype=torch.float64)
    vec[0, 0] = 1.0

    # Accumulate the products of Jacobians (Tn) acting on U0
    for t in range(M):
        next_vec = torch.zeros_like(vec)
        next_vec[0, 0] = torch.sum(dy_dxlag_sub[t] * vec.squeeze())
        if m > 1:
            # Shift the identity elements (J(2:end, 1:end-1) = eye(m-1))
            next_vec[1:, 0] = vec[:-1, 0]
        vec = next_vec

    # vec is now Tn(M) * U0
    # Calculate dominant Lyapunov Exponent
    v = torch.sum(vec**2)  # Equivalent to (Tn * U0)' * (Tn * U0)
    v = torch.clamp(v, min=1e-300)  # prevent log(0)

    if torch.isinf(v) or torch.isnan(v):
        return float('-inf')

    lambda_exp = (1.0 / (2.0 * M)) * torch.log(v)
    return lambda_exp.item()


def fit_pytorch_two_stage(XLAG, target, L, m, q, active_fn, n_starts, device):
    S = int(n_starts)
    g = torch.Generator(device=device).manual_seed(42 + L * 100 + m * 10 + q)

    B = (torch.randn((S, q, m), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    C = (torch.randn((S, 1, q), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    A = (torch.randn((S, q, 1), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    D = (torch.randn((S, 1, 1), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()

    XLAG_batch = XLAG.unsqueeze(0)
    target_batch = target.unsqueeze(0)

    # --- Stage 1: Extended fast parallel exploration ---
    adam = torch.optim.Adam([B, C, A, D], lr=0.03)
    for _ in range(1000):  # INCREASED from 350
        adam.zero_grad(set_to_none=True)
        u = torch.matmul(XLAG_batch, B.transpose(1, 2)) + C
        f_u = activation_fn_torch(u, active_fn)
        y_fit = (D + torch.matmul(f_u, A)).squeeze(-1)
        loss = torch.sum((target_batch - y_fit) ** 2)
        loss.backward()
        adam.step()

    with torch.no_grad():
        u_stage1 = torch.matmul(XLAG_batch, B.transpose(1, 2)) + C
        f_stage1 = activation_fn_torch(u_stage1, active_fn)
        y_stage1 = (D + torch.matmul(f_stage1, A)).squeeze(-1)
        sse_stage1 = torch.sum((target_batch - y_stage1) ** 2, dim=1)

        valid_mask = torch.isfinite(sse_stage1)
        if not torch.any(valid_mask):
            return float('inf'), None

        sse_clean = torch.where(valid_mask, sse_stage1, torch.tensor(float('inf'), device=device))
        best_idx = int(torch.argmin(sse_clean).item())
        best_stage1_sse = sse_clean[best_idx].item()

    b = B[best_idx].detach().clone().requires_grad_(True)
    c = C[best_idx, 0].detach().clone().requires_grad_(True)
    a = A[best_idx, :, 0].detach().clone().requires_grad_(True)
    d = D[best_idx, 0].detach().clone().requires_grad_(True)

    best_params = (b.detach().cpu().clone(), c.detach().cpu().clone(),
                   a.detach().cpu().clone(), d.detach().cpu().clone())
    best_resnorm = best_stage1_sse

    # --- Stage 2: Second-order L-BFGS polish on the single best start ---
    lbfgs = torch.optim.LBFGS(
        [b, c, a, d],
        max_iter=1000,  # INCREASED from 60
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        u_s = XLAG @ b.T + c
        f_s = activation_fn_torch(u_s, active_fn)
        y_s = d + f_s @ a
        loss_s = torch.sum((target - y_s) ** 2)
        loss_s.backward()
        return loss_s

    try:
        lbfgs.step(closure)
        with torch.no_grad():
            u_final = XLAG @ b.T + c
            f_final = activation_fn_torch(u_final, active_fn)
            y_final = d + f_final @ a
            resnorm = torch.sum((target - y_final) ** 2).item()

        if np.isfinite(resnorm) and resnorm <= best_resnorm:
            best_resnorm = resnorm
            best_params = (b.detach().cpu().clone(), c.detach().cpu().clone(),
                           a.detach().cpu().clone(), d.detach().cpu().clone())
    except Exception:
        pass

    return best_resnorm, best_params


def reconstruct_and_plot_pytorch(X_tensor, params, L, m, q, active_fn='tanh', save_path='reconstruction.png'):
    B, C, A, D = params
    XLAG = lag_matrix_torch(X_tensor, L, m)

    with torch.no_grad():
        u = XLAG @ B.T + C
        f_u = activation_fn_torch(u, active_fn)
        y_fit = (D + f_u @ A).numpy()

    target = X_tensor[m * L:].numpy()
    t_axis = np.arange(m * L, len(X_tensor))

    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, target, label='Original Series $X_t$', color='black', alpha=0.8)
    plt.plot(t_axis, y_fit, label=f'Reconstructed PyTorch ($L={L}, m={m}, q={q}$)', color='red', linestyle='--')
    plt.xlabel('Time Step $t$')
    plt.ylabel('$X_t$')
    plt.title('Original vs. PyTorch Reconstructed Time Series')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return y_fit


if __name__ == '__main__':
    log_file = f"chaostest_GPU_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)
    logger.info(f"Saving execution logs to: {log_file}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        mat_file = scipy.io.loadmat('logistic_map.mat')
        X_cpu = torch.from_numpy(mat_file['x']).squeeze().float()
    except FileNotFoundError:
        logger.warning("logistic_map.mat not found. Generating dummy test data...")
        x_np = np.zeros(2000)
        x_np[0] = 0.5
        for i in range(1, 2000):
            x_np[i] = 3.9 * x_np[i-1] * (1 - x_np[i-1])
        X_cpu = torch.from_numpy(x_np).float()

    X_gpu = X_cpu.to(device)

    maxL, maxM, maxQ = 2, 3, 6
    active_fn = 'tanh'

    # SIGNIFICANT IMPROVEMENT:
    # Leverage GPU massively to brute-force a better global minimum before L-BFGS
    n_starts = 1024

    total_tasks = maxL * maxM * maxQ
    best_lambda = float('-inf')
    best_resnorm_global = float('inf')
    best_orders = None
    best_params = None

    logger.info(f"Starting GPU optimization across {total_tasks} combinations ({n_starts} parallel starts) on {device}...")
    start_time = time.time()

    completed = 0
    for L in range(1, maxL + 1):
        for m in range(1, maxM + 1):
            XLAG = lag_matrix_torch(X_gpu, L, m).to(device)
            target = X_gpu[m * L:]

            for q in range(1, maxQ + 1):
                completed += 1

                # 1. Fit the model to minimize SSE
                resnorm, params = fit_pytorch_two_stage(XLAG, target, L, m, q, active_fn, n_starts, device)

                if params is None:
                    elapsed = time.time() - start_time
                    logger.warning(f"[{completed:3d}/{total_tasks:3d}] (L={L}, m={m}, q={q}) | Optimization diverged (NaN/Inf) | Elapsed: {elapsed:.1f}s")
                    continue

                # 2. Compute Lyapunov Exponent from the fitted parameters
                B, C, A, D = [p.to(device) for p in params]
                lambda_exp = compute_lyapunov(XLAG, B, C, A, active_fn)

                elapsed = time.time() - start_time

                logger.info(f"[{completed:3d}/{total_tasks:3d}] (L={L}, m={m}, q={q}) | SSE = {resnorm:.4g} | lambda = {lambda_exp:+.4f} | Elapsed: {elapsed:.1f}s")

                # 3. Model selection criteria: Maximum Lyapunov Exponent
                change = False
                if best_orders is None:
                    change = True
                elif lambda_exp > best_lambda:
                    change = True
                elif lambda_exp == best_lambda and (L + m + q) < sum(best_orders):
                    change = True

                if change:
                    best_lambda = lambda_exp
                    best_resnorm_global = resnorm
                    best_orders = (L, m, q)
                    best_params = params
                    logger.info(f"  └─> New best lambda: {best_lambda:+.4f} at triplet (L={L}, m={m}, q={q}) (SSE={resnorm:.4g})")

    total_time = time.time() - start_time
    logger.info("")
    logger.info(f"Done in {total_time:.1f}s!")
    logger.info(f"PyTorch Best Orders (L, m, q): {best_orders} with lambda = {best_lambda:+.4f} (SSE = {best_resnorm_global:.4f})")

    if best_params is not None:
        L_best, m_best, q_best = best_orders
        y_reconstructed = reconstruct_and_plot_pytorch(X_cpu, best_params, L_best, m_best, q_best, active_fn)
        logger.info("Saved reconstruction plot to: reconstruction.png")
