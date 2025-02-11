import torch


t: torch.Tensor = torch.Tensor([2, -1, 0])
m: torch.Tensor = torch.Tensor([0, 0, 0])
M: torch.Tensor = torch.Tensor([1, 1, 1])

print(torch.clamp(t, m, M))


tensor = torch.tensor([[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [2, 1, 1, 1]], dtype=torch.float32)

print(tensor)

print(torch.max(tensor, dim=0)[0])