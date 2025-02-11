from typing import Union
import pandas as pd
import os
from src.datasets.dataset import DataInfo
from ..metrics.metrics import perturbation_distance, graph_edit_distance, fidelity, sample_distance_from_mean, node_sparsity, edge_sparsity, sample_distance_from_mean_projection, factual_counterfactual_distance
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import numpy as np

name_dict = {"CF-GNNExplainer": "CF-GNN", 
            "CF-GNNExplainer Features": "CF-GNNF",
            "RandomPerturbation": "RND",
            "RandomFeatures": "RNDF"}

def compute_metrics(factual: Data, counterfactual: Union[Data, None], data_info: DataInfo, device: str = "cpu", time: float = 0.0) -> dict:

    if counterfactual is None:
        return {
                "Perturbation Distance": torch.nan,
                "GED": torch.nan,
                "Distribution Distance": torch.nan,
                "Distribution Distance Projection": torch.nan,
                "Fidelity": torch.nan,
                "Counterfactual Distance": torch.nan,
                "Node Sparsity": torch.nan,
                "Edge Sparsity": torch.nan,
                "Time": torch.nan,                
                "Validity": False}

    else:
        
        factual = factual.to(device)
        counterfactual = counterfactual.to(device)
        
        return {
                "Perturbation Distance": perturbation_distance(factual, counterfactual),
                "GED": graph_edit_distance(factual, counterfactual),
                "Distribution Distance": sample_distance_from_mean(data_info.distribution_mean.to(device), counterfactual),
                "Distribution Distance Projection": sample_distance_from_mean_projection(data_info.distribution_mean_projection.to(device), counterfactual),
                "Fidelity": fidelity(factual, counterfactual),
                "Counterfactual Distance": factual_counterfactual_distance(factual, counterfactual),
                "Node Sparsity": node_sparsity(factual, counterfactual),
                "Edge Sparsity": edge_sparsity(factual, counterfactual),
                "Time": time,
                "Validity": True}

