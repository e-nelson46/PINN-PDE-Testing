import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1,8)
        self.hidden_layer2 = nn.Linear(8,16)
        self.output_layer = nn.Linear(16,1)

    def forward(self, x):
        inputs = torch.tensor(x)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out)

        return output

#x = Net()
#net = net.to(device)

x = torch.arange(1,21, requires_grad=True, dtype=torch.float32)
u = torch.sin(x)

du = torch.autograd.grad(u, x, create_graph=True,  grad_outputs=torch.ones_like(u))[0]
duu = torch.autograd.grad(du, x, create_graph=True,  grad_outputs=torch.ones_like(du))[0]

print(f"Original {u}")
print(f"1st Deriv {du}")
print(f"2nd Deriv {duu}")