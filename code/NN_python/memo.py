import torch
from torch.func import jacrev
torch.set_default_dtype(torch.float64)

# calculer la dérivée de tanh(x) évaluée pour x=0.8
x = torch.tensor(0.8, requires_grad = True)
y = torch.tanh(x)
a = y.backward()
z = x.grad

#Calcul de Jacobienne
def f(v):
    return torch.stack([v[0]*v[1],torch.tanh(v[0])])

print(jacrev(f)(torch.tensor([0.8,2.0])))
