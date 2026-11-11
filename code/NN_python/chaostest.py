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
    L, m, q, X_tensor, active_fn, start_value = args

    XLAG = lag_matrix_torch(X_tensor, L, m)
    target = X_tensor[m * L:]

    # Initialize explicit parameter tensors with gradients enabled
    B = torch.full((q, m), start_value, dtype=torch.float32, requires_grad=True)
    C = torch.full((q,), start_value, dtype=torch.float32, requires_grad=True)
    A = torch.full((q,), start_value, dtype=torch.float32, requires_grad=True)
    D = torch.full((1,), start_value, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [B, C, A, D],
        max_iter=100,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        # Evaluate model: u = XLAG @ B.T + C -> y_fit = D + f(u) @ A
        u = XLAG @ B.T + C
        f_u = activation_fn_torch(u, active_fn)
        y_fit = D + f_u @ A

        # Sum of Squared Errors (SSE) loss
        loss = torch.sum((target - y_fit)**2)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Compute final sum of squared errors
    with torch.no_grad():
        u_final = XLAG @ B.T + C
        f_u_final = activation_fn_torch(u_final, active_fn)
        y_fit_final = D + f_u_final @ A
        resnorm = torch.sum((target - y_fit_final)**2).item()

    if np.isnan(resnorm) or np.isinf(resnorm):
        resnorm = float('inf')

    # Return detached parameter copies
    params = (B.detach().clone(), C.detach().clone(), A.detach().clone(), D.detach().clone())
    return (resnorm, (L, m, q), params)

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

    start_value = 0.5
    maxL, maxM, maxQ = 5, 6, 5
    active_fn = 'tanh'

    tasks = [
        (L, m, q, X_tensor, active_fn, start_value)
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
