import torch
from models import activation_fn
from utils import load_mat

def test_activation():
    m, q = 3, 2  # m embedding dimension, q layers
    mat_file = load_mat('/home/gaspichou/Documents/git/chaos_from_noise/database/generated_sig/logistic_map.mat')
    timeseries = torch.from_numpy(mat_file['x']).squeeze()
    # Test datas
    x_lags = torch.randn(m, dtype=torch.float64)
    A = torch.ones(q, dtype=torch.float64)
    B = torch.ones((q, m), dtype=torch.float64)
    C = torch.ones(q, dtype=torch.float64)
    alpha_0 = torch.tensor(0.5, dtype=torch.float64)


    out = activation_fn(x_lags, A, B, C, alpha_0)

    # Check dim
    print(f"Size of the output : {out.shape}")
    print(f"Computed value    : {out.item():.6f}")
    assert out.ndim == 0, "Output needs to be a scalar"
    print("Test passed !")

if __name__ == "__main__":
    test_activation()
