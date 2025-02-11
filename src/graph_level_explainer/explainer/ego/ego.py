import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from ....abstract.explainer import Explainer
from ...utils.utils import build_counterfactual_graph_gc
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

class EgoExplainer(Explainer):
	
    def __init__(self, cfg:DictConfig, datainfo) -> None:
        super().__init__(cfg=cfg, datainfo=datainfo)	

        self.set_reproducibility()
        

    def explain(self, graph: Data, oracle) -> dict:

        _, ego_edge_index, _, _ = k_hop_subgraph(node_idx=0, num_hops=1, edge_index=graph.edge_index)
        out = oracle(graph.x, ego_edge_index, graph.batch)
        out_original = oracle(graph.x, graph.edge_index, graph.batch)
        pred_cf = torch.argmax(out, dim=1)
        pred_orig = torch.argmax(out_original, dim=1)

        counterfactual = None
      
        if pred_cf != pred_orig:

            counterfactual = build_counterfactual_graph_gc(x=graph.x, 
                                                        edge_index=ego_edge_index, 
                                                        graph=graph, 
                                                        oracle=oracle, 
                                                        output_actual=out)

        return counterfactual

    @property
    def name(self):
        
        return "EGO" 