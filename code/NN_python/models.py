import torch

def activation_fn_torch(u, fn_type='tanh'):
    fn_type = fn_type.lower()
    if fn_type == 'tanh':
        return torch.tanh(u)
    elif fn_type == 'logistic':
        return 1.0 / (1.0 + torch.exp(-u))
    elif fn_type == 'sigmoid':
        return u * (1.0 + torch.abs(u / 2.0)) / (2.0 + torch.abs(u) + (u**2) / 2.0)
    else:
        raise ValueError(f"Unknown activation function: {fn_type}")
