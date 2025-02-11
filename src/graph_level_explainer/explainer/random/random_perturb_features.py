import time
from omegaconf import DictConfig
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn.functional as F
from ....abstract.explainer import Explainer
from ...utils.utils import build_counterfactual_graph_gc
from torch import nn

class RandomFeaturesExplainer(Explainer):
    
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

        best_loss = np.inf
        counterfactual = None
        
        start = time.time()
        
        for _ in range(self.cfg.explainer.epochs):
            # Generate random perturbations for the node features
            P_c = torch.empty_like(graph.x).uniform_(0, 1).to(self.device)
            
            P_c = P_c * (self.datainfo.max_range.to(self.device) - self.datainfo.min_range.to(self.device)) + self.datainfo.min_range.to(self.device)
            
            P_d = torch.empty_like(graph.x).uniform_(0, 1).to(self.device)
            
            P_d = P_d * (self.datainfo.max_range.to(self.device) - self.datainfo.min_range.to(self.device)) + self.datainfo.min_range.to(self.device)

            P_d = torch.round(P_d)
            
            P_x = P_d * self.datainfo.discrete_mask.to(self.device) + P_c * (1 - self.datainfo.discrete_mask.to(self.device))
            
            # Apply perturbations and clamp the values between 0 and 1
            perturbed_features = torch.clamp(P_x + graph.x, self.datainfo.min_range.to(self.device), self.datainfo.max_range.to(self.device))
            
            # Get the prediction for the perturbed features
            out = oracle(perturbed_features, graph.edge_index, graph.batch)
            
            # Get the predicted class for the counterfactual
            pred_cf = torch.argmax(out, dim=1)

            # Calculate the prediction loss and feature loss
            loss_pred = F.cross_entropy(out, graph.targets.unsqueeze(0))
            loss_feat = F.l1_loss(graph.x, perturbed_features)

            # Total loss is the sum of prediction loss and feature loss
            loss_tot = loss_feat + loss_pred
                
            # Check if the counterfactual prediction matches the target and has a lower loss
            if (pred_cf == graph.targets) and (loss_tot.item() < best_loss):
                counterfactual = build_counterfactual_graph_gc(x=perturbed_features, 
                                                            edge_index=graph.edge_index, 
                                                            graph=graph, 
                                                            oracle=oracle, 
                                                            output_actual=out)
                best_loss = loss_tot.item()

            # Check if the timeout has been reached
            if time.time() - start > self.cfg.timeout:
                return counterfactual
            
        return counterfactual

    @property
    def name(self):
        
        return "RandomFeatures" 