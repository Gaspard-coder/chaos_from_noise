import torch
import scipy.io
from models import activation_fn
from utils import lag_matrix


def chaostest(maxL,maxM,maxQ,alpha):
    mat_file = scipy.io.loadmat('/home/gaspichou/Documents/git/chaos_from_noise/database/generated_sig/logistic_map.mat')
    timeseries = torch.from_numpy(mat_file['x']).squeeze() #squeeze from (N,1) matrix to (N,) vector

    # Initialize values
    start_value = 0.5
    alpha_0 = torch.tensor(start_value)
    lambda_val = - torch.inf #lambda is a forbidden word in python


    for l in range(1,maxL+1):
        for m in range(1,maxM+1):
            for q in range(1,maxQ+1):
                # Coefficient matrix initilization
                A = torch.full((q,1),start_value) # (q,1) matrix
                B = torch.full((q,m),start_value) # (q,m) matrix
                C = torch.full((q,1),start_value) # (q,1) matrix

                Theta = [A.flatten(),B,C.flatten(),alpha_0]
