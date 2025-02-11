import torch
from torch_geometric.nn import GCNConv

# Define edge indices
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

# Define node features
x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)

# Define edge weights
edge_weight = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float)

# Initialize GCNConv layer
conv = GCNConv(in_channels=1, out_channels=1, bias=False)

# Perform the forward pass
out_edge_index = conv(x, edge_index, edge_weight)
print("Output using edge_index representation:")
print(out_edge_index)

# Extract the weight from GCNConv
weight = conv.lin.weight

# Define the adjacency matrix
adj = torch.tensor([[0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]], dtype=torch.float)

# Add self-loops
adj = adj.t() + torch.eye(adj.size(0))

# Compute the degree matrix and its inverse square root
degree = adj.sum(dim=1)
degree_inv_sqrt = degree.pow(-0.5)
degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
d_mat_inv_sqrt = torch.diag(degree_inv_sqrt)

# Normalize the adjacency matrix
adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

# Apply the GCN layer using the full adjacency matrix
out_full_adj = adj_normalized @ x @ weight.t()
print("Output using full adjacency matrix representation:")
print(out_full_adj)
