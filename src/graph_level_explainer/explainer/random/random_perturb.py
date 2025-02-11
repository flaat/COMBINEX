import time
from omegaconf import DictConfig
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from ....abstract.explainer import Explainer
from ...utils.utils import build_counterfactual_graph_gc
from torch import Tensor
from torch import nn



class RandomExplainer(Explainer):

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

        # Parameters initialization
        counterfactual = None
        best_loss = np.inf
        
        start = time.time()

        for _ in range(self.cfg.explainer.epochs):
            
            P_e = torch.randint(low=0, high=2, size=(len(graph.edge_index[0]), ), device=self.device).float()
            
            cf_edge_index = graph.edge_index[:, P_e == 1]

            out = oracle(graph.x, graph.edge_index, graph.batch, P_e)
            
            pred_cf = torch.argmax(out, dim=1)            
            
            # Counterfactual fidelity loss (e.g., cross-entropy)
            fidelity_loss = F.cross_entropy(out, graph.targets.unsqueeze(0))

            # Graph sparsity loss: Penalize large changes in edge weights
            sparsity_loss = torch.sum(torch.abs(P_e - 1))  # Penalize deviations from original weights

            # Combine losses: fidelity + sparsity regularization
            total_loss = fidelity_loss + sparsity_loss
            
            if (pred_cf.item() == graph.targets.item()) and (total_loss.item() < best_loss):

                counterfactual = build_counterfactual_graph_gc(x=graph.x, 
                                                            edge_index=cf_edge_index, 
                                                            graph=graph, 
                                                            oracle=oracle, 
                                                            output_actual=out)                
                best_loss = total_loss
                
            if time.time() - start > self.cfg.timeout:
            
                return counterfactual

        return counterfactual
    
    @property
    def name(self):
        
        return "RandomPerturbation" 