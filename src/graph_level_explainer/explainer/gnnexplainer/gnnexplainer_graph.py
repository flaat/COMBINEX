import time
from omegaconf import DictConfig
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.explain import  GNNExplainer
from torch_geometric.explain import  Explainer

from ....abstract.explainer import ExplainerABC 
from ...utils.utils import build_counterfactual_graph_gc
from torch import nn

class GNNExplainerWrap(ExplainerABC):
    
    def __init__(self, cfg:DictConfig, datainfo) -> None:
        super().__init__(cfg=cfg, datainfo=datainfo)

        self.set_reproducibility()


    def explain(self, graph: Data, oracle: nn.Module) -> dict:
        """
        Explain a node by perturbing its features randomly and evaluating the impact on the model's prediction.

        Args:
            graph (Data): The input graph data containing node features and edge indices.
            oracle (Callable): The model used to make predictions on the graph.

        Returns:
            dict: A dictionary containing the counterfactual graph if found, otherwise None.
        """
        # Get the original prediction from the oracle

        counterfactual = None
        
        start = time.time()
        
        explainer = Explainer(
            model=oracle,
            algorithm=GNNExplainer(epochs=1000),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='log_probs',
            ),
        )
        # Get prediction for original graph
        original_out = oracle(graph.x, graph.edge_index, graph.batch)
        original_pred = original_out.argmax(dim=1)

        # Generate explanation using GNNExplainer
        explanation = explainer(graph.x, graph.edge_index)

        print("Original Prediction:", original_pred)

        # Identify the top-k important edges to remove (based on edge mask values)
        k = 5  # Number of edges to remove for counterfactual generation
        important_edges = explanation.edge_mask.argsort(descending=True)[:k]

        # Remove the most important edges to create a counterfactual graph
        edge_index = graph.edge_index.clone()
        for i in important_edges:
            edge_index[:, i] = -1  # Mark edge for removal

        # Filter out removed edges
        edge_index = edge_index[:, edge_index[0] != -1]

        # Perturb node features based on node mask (first 4 elements only)
        node_features = graph.x.clone()
        node_mask = explanation.node_mask

        # Apply perturbation to the top-k most important node features
        for node_idx in range(node_features.shape[0]):
            # Perturb the first 4 elements based on importance from node mask
            important_features = node_mask[node_idx, :4].argsort(descending=True)[:k % 4]
            for feature_idx in important_features:
                # Randomly perturb the feature value to simulate a counterfactual change
                node_features[node_idx, feature_idx] = torch.rand(1).item() * (1 - node_features[node_idx, feature_idx])

        # Create the counterfactual graph data
        counterfactual_data = graph.clone()
        counterfactual_data.edge_index = edge_index
        counterfactual_data.x = node_features

        # Forward pass with modified graph to check if the prediction changes
        counterfactual_out = oracle(counterfactual_data.x, counterfactual_data.edge_index, graph.batch)
        counterfactual_pred = counterfactual_out.argmax(dim=1)

        # Store the counterfactual results
        counterfactual = Data(x=counterfactual_data.x, edge_index=counterfactual_data.edge_index, y=counterfactual_pred)
        
        # Check if the counterfactual prediction matches the target and has a lower loss
        if (counterfactual_pred == graph.targets):
            counterfactual = build_counterfactual_graph_gc(x=node_features, edge_index=graph.edge_index, graph=graph, oracle=oracle, output_actual=counterfactual_out)
            return counterfactual
        
        # Check if the timeout has been reached
        if time.time() - start > self.cfg.timeout:
            return counterfactual
        
        return None
        
        

    @property
    def name(self):
        
        return "GNNExplainer" 