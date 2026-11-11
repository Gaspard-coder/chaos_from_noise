import torch

def lag_matrix_torch(X, L, m):
    """
    Creates the T x m matrix of lagged observations X(t - i*L).
    T = len(X) - m*L
    """
    N = len(X)
    T = N - m * L
    XLAG = torch.zeros((T, m), dtype=X.dtype)
    for i in range(1, m + 1):
        XLAG[:, i - 1] = X[(m * L - i * L):(N - i * L)]
    return XLAG
