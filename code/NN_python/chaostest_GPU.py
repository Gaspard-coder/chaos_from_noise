import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
import time

from utils import lag_matrix_torch
from models import activation_fn_torch


def fit_pytorch_two_stage(XLAG, target, L, m, q, active_fn, n_starts, device):
    """
    Stage 1: Optimize `n_starts` initializations independently in parallel using batched Adam.
    Stage 2: Polish the best candidate with single-trajectory L-BFGS for machine-precision SSE.
    """
    S = int(n_starts)
    g = torch.Generator(device=device).manual_seed(42 + L * 100 + m * 10 + q)

    B = (torch.randn((S, q, m), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    C = (torch.randn((S, 1, q), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    A = (torch.randn((S, q, 1), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    D = (torch.randn((S, 1, 1), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_()

    XLAG_batch = XLAG.unsqueeze(0)      # (1, N_eff, m)
    target_batch = target.unsqueeze(0)  # (1, N_eff)

    # --- Stage 1: Fast parallel exploration (no CPU-GPU line-search syncs) ---
    adam = torch.optim.Adam([B, C, A, D], lr=0.03)
    for _ in range(350):
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

    # Extract winning candidate as standalone leaf tensors for L-BFGS
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
        max_iter=60,
        tolerance_grad=1e-6,
        tolerance_change=1e-8,
        history_size=20,
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
    plt.show()

    return y_fit


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mat_file = scipy.io.loadmat('logistic_map.mat')
    X_cpu = torch.from_numpy(mat_file['x']).squeeze().float()
    X_gpu = X_cpu.to(device)

    maxL, maxM, maxQ = 2, 3, 6
    active_fn = 'tanh'
    n_starts = 32  # 32 parallel starts in Stage 1, polished by L-BFGS in Stage 2

    total_tasks = maxL * maxM * maxQ
    best_resnorm = float('inf')
    best_orders = None
    best_params = None

    print(f"Starting 2-stage GPU optimization across {total_tasks} combinations ({n_starts} parallel starts) on {device}...")
    start_time = time.time()

    completed = 0
    for L in range(1, maxL + 1):
        for m in range(1, maxM + 1):
            XLAG = lag_matrix_torch(X_gpu, L, m).to(device)
            target = X_gpu[m * L:]

            for q in range(1, maxQ + 1):
                completed += 1
                resnorm, params = fit_pytorch_two_stage(XLAG, target, L, m, q, active_fn, n_starts, device)
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
