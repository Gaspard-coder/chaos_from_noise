import torch

def lag_matrix(timeseries,m,L):
    '''
    Extract [x_{t_L},x_{t_2L},...,x_{t_mL}] from the original timeseries
    '''

    T = timeseries.shape[0]
    if T <=m*L:
        raise ValueError(f"Timeseries is too short, needs at least m*L points")

    # t_indices (T - m*L,1),
    t_index = torch.arange(m * L, T, device=timeseries.device).unsqueeze(1)
    lags = (torch.arange(1,m+1,device=timeseries.device)*L).unsqueeze(1)

    idx_matrix = t_index - lags
    # Column k = x_{t-k*L}
    return timeseries[idx_matrix]
