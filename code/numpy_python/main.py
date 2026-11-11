import numpy as np
import scipy.io
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def lag_matrix(X, L, m):
    """
    Creates the T x m matrix of lagged observations X(t - i*L).
    T = len(X) - m*L
    """
    N = len(X)
    T = N - m * L
    XLAG = np.zeros((T, m))
    for i in range(1, m + 1):
        # Column i-1 contains X lagged by i*L
        XLAG[:, i - 1] = X[(m * L - i * L):(N - i * L)]
    return XLAG

def activation_fn(u, fn_type='tanh'):
    fn_type = fn_type.lower()
    if fn_type == 'tanh':
        return np.tanh(u)
    elif fn_type == 'logistic':
        return 1.0 / (1.0 + np.exp(-u))
    elif fn_type == 'sigmoid':
        return u * (1.0 + np.abs(u / 2.0)) / (2.0 + np.abs(u) + (u**2) / 2.0)
    else:
        raise ValueError(f"Unknown activation function: {fn_type}")

def chaosfn_residual(Theta, X, L, m, q, ActiveFN='tanh'):
    """
    Objective function returning the residual vector (Target - Fitted).
    """
    # Unpack Theta according to the paper's matrix structure:
    # B: (q, m), C: (q,), A: (q,), D: scalar
    B = Theta[:q * m].reshape(q, m)
    C = Theta[q * m : q * m + q]
    A = Theta[q * m + q : q * m + 2 * q]
    D = Theta[q * m + 2 * q]

    # Generate lag matrix XLAG (T, m)
    XLAG = lag_matrix(X, L, m)

    # Calculate u = B * XLAG' + C  ->  shape (T, q)
    u = XLAG @ B.T + C

    # Evaluate model: y_fit = D + sum(A_j * f(u_j))
    y_fit = D + activation_fn(u, ActiveFN) @ A

    # Residual: target time series X(m*L+1:end) - fitted values
    target = X[m * L:]
    return target - y_fit

def fit_model(args):
    L, m, q, X, active_fn, start_value = args
    B_0 = np.full((q, m), start_value)
    C_0 = np.full((q,), start_value)
    A_0 = np.full((q,), start_value)
    Theta0 = np.concatenate([B_0.flatten(), C_0, A_0, [start_value]])

    res = least_squares(
        chaosfn_residual,
        Theta0,
        args=(X, L, m, q, active_fn),
        method='lm'
    )

    resnorm = np.sum(res.fun**2)
    if np.isnan(resnorm) or np.isinf(resnorm):
        resnorm = np.inf

    return (resnorm, (L, m, q), res.x)

def reconstruct_and_plot(X, Theta, L, m, q, active_fn='tanh'):
    """
    Reconstructs the series from the optimized parameters and plots
    the original versus reconstructed signal.
    """
    B = Theta[:q * m].reshape(q, m)
    C = Theta[q * m : q * m + q]
    A = Theta[q * m + q : q * m + 2 * q]
    D = Theta[q * m + 2 * q]

    XLAG = lag_matrix(X, L, m)
    u = XLAG @ B.T + C
    y_fit = D + activation_fn(u, active_fn) @ A

    target = X[m * L:]
    t_axis = np.arange(m * L, len(X))

    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, target, label='Original Series $X_t$', color='black', alpha=0.8)
    plt.plot(t_axis, y_fit, label=f'Reconstructed Neural Net ($L={L}, m={m}, q={q}$)', color='red', linestyle='--')
    plt.xlabel('Time Step $t$')
    plt.ylabel('$X_t$')
    plt.title('Original vs. Reconstructed Time Series')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return y_fit

if __name__ == '__main__':
    mat_file = scipy.io.loadmat('/home/gaspichou/Documents/git/chaos_from_noise/database/generated_sig/logistic_map.mat')
    X = mat_file['x'].squeeze()

    start_value = 0.5
    maxL, maxM, maxQ = 5, 6, 5
    active_fn = 'tanh'

    tasks = [
        (L, m, q, X, active_fn, start_value)
        for L in range(1, maxL + 1)
        for m in range(1, maxM + 1)
        for q in range(1, maxQ + 1)
    ]
    total_tasks = len(tasks)

    best_resnorm = np.inf
    best_orders = None
    best_theta = None

    print(f"Starting Scipy optimization across {total_tasks} parameter combinations...")
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(fit_model, task): task for task in tasks}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            resnorm, orders, theta = future.result()
            L, m, q = orders
            elapsed = time.time() - start_time

            print(f"[{completed:3d}/{total_tasks:3d}] (L={L}, m={m}, q={q}) | SSE = {resnorm:.6f} | Elapsed: {elapsed:.1f}s")

            if resnorm < best_resnorm:
                best_resnorm = resnorm
                best_orders = orders
                best_theta = theta
                print(f"  └─> New Best SSE: {best_resnorm:.6f} at triplet (L={L}, m={m}, q={q})")

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f}s!")
    print(f"Best Orders (L, m, q): {best_orders} with SSE = {best_resnorm:.6f}")

    L_best, m_best, q_best = best_orders
    y_reconstructed = reconstruct_and_plot(X, best_theta, L_best, m_best, q_best, active_fn)
