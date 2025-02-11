import unittest
import torch
import numpy as np
from torch_geometric.data import Data
from torch import nn, Tensor
from src.node_level_explainer.utils.utils import (
    get_degree_matrix,
    create_symm_matrix_from_vec,
    create_vec_from_symm_matrix,
    index_to_mask,
    check_graphs,
    discretize_tensor,
    discretize_to_nearest_integer
)

class TestUtils(unittest.TestCase):

    def test_get_degree_matrix(self):
        adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        degree_matrix = get_degree_matrix(adj)
        expected = torch.diag(torch.tensor([1, 2, 1], dtype=torch.float32))
        self.assertTrue(torch.equal(degree_matrix, expected))


    def test_index_to_mask(self):
        index = torch.tensor([0, 2], dtype=torch.long)
        size = 4
        mask = index_to_mask(index, size)
        expected = torch.tensor([True, False, True, False], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_check_graphs(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.assertFalse(check_graphs(edge_index))
        edge_index_empty = torch.tensor([[], []], dtype=torch.long)
        self.assertTrue(check_graphs(edge_index_empty))

    def test_discretize_tensor(self):
        tensor = torch.tensor([-0.6, -0.5, 0, 0.5, 0.6], dtype=torch.float32)
        discretized = discretize_tensor(tensor)
        expected = torch.tensor([-1, -1, 0, 0, 1], dtype=torch.float32)
        self.assertTrue(torch.equal(discretized, expected))

    def test_discretize_to_nearest_integer(self):
        tensor = torch.tensor([0.1, 0.5, 0.9, 1.5, 9.4, -0.51, -0.5, 0.51], dtype=torch.float32)
        discretized = discretize_to_nearest_integer(tensor)
        expected = torch.tensor([0, 0, 1, 2, 9, -1, 0, 1], dtype=torch.float32)
        self.assertTrue(torch.equal(discretized, expected))

if __name__ == '__main__':
    unittest.main()