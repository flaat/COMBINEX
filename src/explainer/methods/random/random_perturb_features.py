import time
from omegaconf import DictConfig
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn.functional as F
from ....abstract.explainer import ExplainerABC
from ....utils.utils import ExplanationContext, build_counterfactual_graph_gc
from torch import nn

class RandomFeaturesExplainer(ExplainerABC):
    
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
            args, target = None, None
            if self.cfg.task.name.lower() == "node":
                args = {"x":perturbed_features, "edge_index":graph.edge_index, "edge_weights":torch.ones(graph.edge_index.size(1), device=self.device)}
                target = graph.targets[graph.new_idx]
            elif self.cfg.task.name.lower() == "graph":
                args = {"x":perturbed_features, "edge_index":graph.edge_index, "batch":graph.batch,  "edge_weights":torch.ones(graph.edge_index.size(1), device=self.device)}
                target = graph.targets.unsqueeze(0)  # Assuming a single graph in batch
            
            
            
            # Get the prediction for the perturbed features
            out = oracle(**args)
            
            # Get the predicted class for the counterfactual
            pred_cf = torch.argmax(out, dim=1)
            if self.cfg.task.name.lower() == "node":
                out = out[graph.new_idx]

            elif self.cfg.task.name.lower() == "graph":
                out = out
            
            # Calculate the prediction loss and feature loss
            loss_pred = F.cross_entropy(out, target)
            loss_feat = F.l1_loss(perturbed_features, graph.x)

            # Total loss is the sum of prediction loss and feature loss
            loss_tot = loss_feat + loss_pred
            
            context = ExplanationContext(y_pred_new_actual=pred_cf,
                           target=graph.targets,
                           loss=loss_tot,
                           best_loss=self.best_loss,
                           x=perturbed_features,
                           edge_index=graph.edge_index,
                           edge_attr=None,
                           graph=graph,
                           oracle=oracle,
                           output_actual=out,
                           device=self.device)
        
        
            return self.finalize_explanation(context)    


    @property
    def name(self):
        
        return "RandomFeatures" 