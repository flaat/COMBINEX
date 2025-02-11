import sys
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from oracles.models.gcn import GCN
import torch.nn.functional as F

def test_gcn_with_dense_sparse_matrix():
    num_features = 10
    hidden_layers = [16, 32]
    num_classes = 3
    dropout = 0.5
    model = GCN(num_features, hidden_layers, num_classes, dropout)

    # Create a random dense adjacency matrix and feature matrix
    adj = torch.randint(low=0, high=2, size=(5, 5)).float()
    x = torch.rand((5, num_features))

    # Convert dense adjacency matrix to edge_index
    edge_index, _ = dense_to_sparse(adj)

    # Forward pass with edge_index
    output_sparse = model(x, edge_index)

    # Convert edge_index to SparseTensor
    adj_sparse = SparseTensor.from_edge_index(edge_index)

    # Forward pass with SparseTensor
    output_dense = model(x, adj_sparse)

    assert torch.allclose(output_dense, output_sparse, atol=1e-6), "Output of dense and sparse matrix should be the same."

if __name__ == "__main__":
    test_gcn_with_dense_sparse_matrix()
    print("All tests passed.")