import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from utils import lag_matrix_torch
from models import activation_fn_torch

def fit_pytorch_task(args):
    """
    Worker function fitting the chaotic neural network mapping without nn.Module.
    Optimizes B (q x m), C (q,), A (q,), D (scalar) using L-BFGS.
    """
    L, m, q, X_tensor, active_fn, n_starts = args

    XLAG = lag_matrix_torch(X_tensor, L, m)
    target = X_tensor[m * L:]

    best_task_resnorm = float('inf')
    best_task_params = None

    # Multi-start: try different seeds to break symetry
    for start_idx in range(int(n_starts)):
        # One seed per value
        g = torch.Generator().manual_seed(42 + start_idx * 1000 + L * 100 + m * 10 + q)

        # Gaussian init
        B = (torch.randn((q, m), generator=g, dtype=torch.float32) * 0.2).requires_grad_()
        C = (torch.randn((q,), generator=g, dtype=torch.float32) * 0.2).requires_grad_()
        A = (torch.randn((q,), generator=g, dtype=torch.float32) * 0.2).requires_grad_()
        D = (torch.randn((1,), generator=g, dtype=torch.float32) * 0.2).requires_grad_()

        optimizer = torch.optim.LBFGS(
            [B, C, A, D],
            max_iter=150,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            u = XLAG @ B.T + C
            f_u = activation_fn_torch(u, active_fn)
            y_fit = D + f_u @ A
            loss = torch.sum((target - y_fit)**2)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            continue

        with torch.no_grad():
            u_final = XLAG @ B.T + C
            f_u_final = activation_fn_torch(u_final, active_fn)
            y_fit_final = D + f_u_final @ A
            resnorm = torch.sum((target - y_fit_final)**2).item()

        if not np.isnan(resnorm) and not np.isinf(resnorm) and resnorm < best_task_resnorm:
            best_task_resnorm = resnorm
            best_task_params = (B.detach().clone(), C.detach().clone(), A.detach().clone(), D.detach().clone())

    if best_task_params is None:
        best_task_resnorm = float('inf')

    return (best_task_resnorm, (L, m, q), best_task_params)

def reconstruct_and_plot_pytorch(X_tensor, params, L, m, q, active_fn='tanh'):
    """
    Reconstructs the fitted time series from optimal tensor parameters and plots.
    """
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
    plt.show()

    return y_fit

if __name__ == '__main__':
    mat_file = scipy.io.loadmat('/home/gaspichou/Documents/git/chaos_from_noise/database/generated_sig/logistic_map.mat')
    X_tensor = torch.from_numpy(mat_file['x']).squeeze().float()

    maxL, maxM, maxQ = 5, 6, 5
    active_fn = 'tanh'
    n_starts = 5 # Nb of random try per tuple (L,m,q)

    tasks = [
        (L, m, q, X_tensor, active_fn, n_starts)
        for L in range(1, maxL + 1)
        for m in range(1, maxM + 1)
        for q in range(1, maxQ + 1)
    ]
    total_tasks = len(tasks)

    best_resnorm = float('inf')
    best_orders = None
    best_params = None

    print(f"Starting PyTorch L-BFGS optimization across {total_tasks} parameter combinations...")
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(fit_pytorch_task, task): task for task in tasks}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            resnorm, orders, params = future.result()
            L, m, q = orders
            elapsed = time.time() - start_time

            print(f"[{completed:3d}/{total_tasks:3d}] (L={L}, m={m}, q={q}) | SSE = {resnorm:.6f} | Elapsed: {elapsed:.1f}s")

            if resnorm < best_resnorm:
                best_resnorm = resnorm
                best_orders = orders
                best_params = params
                print(f"  └─> New Best SSE: {best_resnorm:.6f} at triplet (L={L}, m={m}, q={q})")

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f}s!")
    print(f"PyTorch Best Orders (L, m, q): {best_orders} with SSE = {best_resnorm:.6f}")

    L_best, m_best, q_best = best_orders
    y_reconstructed = reconstruct_and_plot_pytorch(X_tensor, best_params, L_best, m_best, q_best, active_fn)
