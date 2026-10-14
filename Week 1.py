import torch

x = torch.arange(1,20, requires_grad=True, dtype=torch.float32)
u = torch.sin(x)

du = torch.autograd.grad(u, x, create_graph=True,  grad_outputs=torch.ones_like(u))[0]
duu = torch.autograd.grad(du, x, create_graph=True,  grad_outputs=torch.ones_like(du))[0]

print(f"Original {u}")
print(f"1st Deriv {du}")
print(f"2nd Deriv {duu}")