import torch
import torch.nn as nn


class AlphaModule(nn.Module):
    def __init__(self, num_module, dim, device='cuda'):
        super().__init__()
        self.device = device
        self.num_module = num_module
        self.dim = dim
        self.nets = self.create_module(self.num_module)

    def create_module(self, n_mod):
        net = []
        for _ in range(n_mod):
            module = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.Tanh(),
                nn.Linear(self.dim, 1),
                nn.Tanh()
            )
            module = module.to(self.device)
            net.append(module)
        return net

    def forward(self, x):
        batch, num_ws, dim = x.shape
        assert num_ws == self.num_module and dim == self.dim
        alphas = torch.Tensor(batch, 0).to(self.device)
        for i in range(self.num_module):
            temp = self.nets[i](x[:, i, :])
            alphas = torch.cat([alphas, temp], dim=1)

        return alphas
