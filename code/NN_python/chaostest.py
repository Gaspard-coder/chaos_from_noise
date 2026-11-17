import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
import time

from utils import lag_matrix_torch
from models import activation_fn_torch


def fit_pytorch_batched(XLAG, target, L, m, q, active_fn, n_starts, device):
    """
    Fits all `n_starts` random initializations simultaneously on a single GPU
    by stacking parameters along dimension 0: B (S, q, m), C (S, 1, q), A (S, q, 1), D (S, 1, 1).
    """
    S = int(n_starts)
    g = torch.Generator(device=device).manual_seed(42 + L * 100 + m * 10 + q)

    B = (torch.randn((S, q, m), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    C = (torch.randn((S, 1, q), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    A = (torch.randn((S, q, 1), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    D = (torch.randn((S, 1, 1), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()

    optimizer = torch.optim.LBFGS(
        [B, C, A, D],
        max_iter=100,
        tolerance_grad=1e-5,
        tolerance_change=1e-7,
        history_size=20,
        line_search_fn="strong_wolfe"
    )

    XLAG_batch = XLAG.unsqueeze(0)      # (1, N_eff, m)
    target_batch = target.unsqueeze(0)  # (1, N_eff)

    def closure():
        optimizer.zero_grad()
        u = torch.matmul(XLAG_batch, B.transpose(1, 2)) + C  # (S, N_eff, q)
        f_u = activation_fn_torch(u, active_fn)              # (S, N_eff, q)
        y_fit = (D + torch.matmul(f_u, A)).squeeze(-1)       # (S, N_eff)
        loss = torch.sum((target_batch - y_fit) ** 2)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        pass

    with torch.no_grad():
        u_final = torch.matmul(XLAG_batch, B.transpose(1, 2)) + C
        f_u_final = activation_fn_torch(u_final, active_fn)
        y_fit_final = (D + torch.matmul(f_u_final, A)).squeeze(-1)
        sse_per_start = torch.sum((target_batch - y_fit_final) ** 2, dim=1)  # (S,)

        # Filter out NaNs/Infs before picking the best start
        valid_mask = torch.isfinite(sse_per_start)
        if not torch.any(valid_mask):
            return float('inf'), None

        sse_clean = torch.where(valid_mask, sse_per_start, torch.tensor(float('inf'), device=device))
        best_idx = torch.argmin(sse_clean).item()
        best_resnorm = sse_clean[best_idx].item()

        best_params = (
            B[best_idx].detach().cpu().clone(),             # (q, m)
            C[best_idx].squeeze(0).detach().cpu().clone(),  # (q,)
            A[best_idx].squeeze(-1).detach().cpu().clone(), # (q,)
            D[best_idx].squeeze(0).detach().cpu().clone()   # (1,)
        )

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
    plt.show()

    return y_fit


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mat_file = scipy.io.loadmat('/home/gaspichou/Documents/git/chaos_from_noise/database/generated_sig/logistic_map.mat')
    X_cpu = torch.from_numpy(mat_file['x']).squeeze().float()
    X_gpu = X_cpu.to(device)

    maxL, maxM, maxQ = 2, 3, 6
    active_fn = 'tanh'
    n_starts = 10  # Batched in parallel on GPU—costs virtually the same as n_starts=1

    total_tasks = maxL * maxM * maxQ
    best_resnorm = float('inf')
    best_orders = None
    best_params = None

    print(f"Starting batched PyTorch L-BFGS optimization across {total_tasks} combinations ({n_starts} starts each) on {device}...")
    start_time = time.time()

    completed = 0
    for L in range(1, maxL + 1):
        for m in range(1, maxM + 1):
            # Build XLAG and target once per (L, m) pair instead of recomputing for every q
            XLAG = lag_matrix_torch(X_gpu, L, m)
            target = X_gpu[m * L:]

            for q in range(1, maxQ + 1):
                completed += 1
                resnorm, params = fit_pytorch_batched(XLAG, target, L, m, q, active_fn, n_starts, device)
                elapsed = time.time() - start_time

                print(f"[{completed:3d}/{total_tasks:3d}] (L={L}, m={m}, q={q}) | SSE = {resnorm:.6f} | Elapsed: {elapsed:.1f}s")

                if resnorm < best_resnorm:
                    best_resnorm = resnorm
                    best_orders = (L, m, q)
                    best_params = params
                    print(f"  └─> New Best SSE: {best_resnorm:.6f} at triplet (L={L}, m={m}, q={q})")

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f}s!")
    print(f"PyTorch Best Orders (L, m, q): {best_orders} with SSE = {best_resnorm:.6f}")

    L_best, m_best, q_best = best_orders
    y_reconstructed = reconstruct_and_plot_pytorch(X_cpu, best_params, L_best, m_best, q_best, active_fn)
