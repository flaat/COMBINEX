# Code taken from TODO: add here
#

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from ....abstract.explainer import Explainer
from ...utils.utils import build_counterfactual_graph, get_neighbourhood, normalize_adj
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph, to_dense_adj
from torch_geometric.data import Data

class EgoExplainer(Explainer):
	
    def __init__(self, cfg:DictConfig, datainfo) -> None:
        super().__init__(cfg=cfg, datainfo=datainfo)	

        self.set_reproducibility()
        

    def explain(self, graph: Data, oracle) -> dict:


        ego_nodes, ego_edge_index, _, _ = k_hop_subgraph(graph.new_idx.item(), 1, graph.edge_index)

        out = oracle(graph.x, ego_edge_index)
        out_original = oracle(graph.x, graph.edge_index)
        pred_cf = torch.argmax(out, dim=1)[graph.new_idx]
        
        pred_orig = torch.argmax(out_original, dim=1)[graph.new_idx]


        counterfactual = None
        
            
        if pred_cf == graph.targets[graph.new_idx]:

            counterfactual = build_counterfactual_graph(x=graph.x, edge_index=ego_edge_index, graph=graph, oracle=oracle, output_actual=out)

        return counterfactual

    @property
    def name(self):
        
        return "EGO" 