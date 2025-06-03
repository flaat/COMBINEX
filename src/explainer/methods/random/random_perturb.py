import time
from omegaconf import DictConfig
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from ....abstract.explainer import ExplainerABC
from ....utils.utils import ExplanationContext, build_counterfactual_graph_gc
from torch import Tensor
from torch import nn



class RandomExplainer(ExplainerABC):

    def __init__(self, cfg: DictConfig, datainfo) -> None:
        super().__init__(cfg=cfg, datainfo=datainfo)
        
        self.set_reproducibility()


    def explain(self, graph: Data, oracle: nn.Module) -> Data:
        """
        Generate a counterfactual explanation for a node in the graph.

        Parameters:
        graph (Data): The input graph data.
        oracle (nn.Module): The model used to make predictions.

        Returns:
        Data: The counterfactual graph data.
        """
        args = None

        for _ in range(self.cfg.explainer.epochs):
            
            P_e = torch.randint(low=0, high=2, size=(len(graph.edge_index[0]), ), device=self.device).float()
            
            cf_edge_index = graph.edge_index[:, P_e == 1]
            target = None
            if self.cfg.task.name.lower() == "node":
                args = {"x":graph.x, "edge_index":graph.edge_index, "edge_weights":P_e}
                target = graph.targets[graph.new_idx]
            elif self.cfg.task.name.lower() == "graph":
                args = {"x":graph.x, "edge_index":graph.edge_index, "batch":graph.batch,  "edge_weights":P_e}
                target = graph.targets.unsqueeze(0)  # Assuming a single graph in batch
                
            out = oracle(**args)
            pred_cf = torch.argmax(out, dim=1)            
            
            if self.cfg.task.name.lower() == "node":
                out = out[graph.new_idx]

            elif self.cfg.task.name.lower() == "graph":
                out = out
            
            
            # Counterfactual fidelity loss (e.g., cross-entropy)
            fidelity_loss = F.cross_entropy(out,target)

            # Graph sparsity loss: Penalize large changes in edge weights
            sparsity_loss = torch.sum(torch.abs(P_e - 1))  # Penalize deviations from original weights

            # Combine losses: fidelity + sparsity regularization
            total_loss = fidelity_loss + sparsity_loss
            
            context = ExplanationContext(y_pred_new_actual=pred_cf,
                           target=graph.targets,
                           loss=total_loss,
                           best_loss=self.best_loss,
                           x=graph.x,
                           edge_index=cf_edge_index,
                           edge_attr=None,
                           graph=graph,
                           oracle=oracle,
                           output_actual=out,
                           device=self.device)
        
        
            return self.finalize_explanation(context)
            
    
    @property
    def name(self):
        
        return "RandomPerturbation" 