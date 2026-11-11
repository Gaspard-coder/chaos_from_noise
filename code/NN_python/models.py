import torch

def activation_fn(x_lags: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha_0: torch.Tensor) -> torch.Tensor:
    """
        x_lags : Tensor (m,) containing [x_{t-L}, ..., x_{t-mL}]
        A      : Matrix (q, m) containing beta_{i,j}
        B      : Vector (q,) containing beta_{0,j}
        C      : Vector (q,) containing alpha_j
        alpha_0: constant
        """

    # B(m,q) @ x_lags (m,) => matrix product leading to (q,)
    u = B @ x_lags + C
    # tanh optimized activation function for chaotic noise timeseries, can be changed
    psi = torch.tanh(u)
    # scalar product A(q,) and psi(q,)
    y = torch.dot(A,psi) + alpha_0
    return y
