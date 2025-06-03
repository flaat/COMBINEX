import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from ....abstract.explainer import ExplainerABC
from ....utils.utils import ExplanationContext, build_counterfactual_graph_gc
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

class EgoExplainer(ExplainerABC):
	
    def __init__(self, cfg:DictConfig, datainfo) -> None:
        super().__init__(cfg=cfg, datainfo=datainfo)	

        self.set_reproducibility()
        

    def explain(self, graph: Data, oracle) -> dict:

        _, ego_edge_index, _, _ = k_hop_subgraph(node_idx=0, num_hops=1, edge_index=graph.edge_index)
        out = oracle(graph.x, ego_edge_index, graph.batch)
        out_original = oracle(graph.x, graph.edge_index, graph.batch)
        pred_cf = torch.argmax(out, dim=1)
        pred_orig = torch.argmax(out_original, dim=1)

        context = ExplanationContext(y_pred_new_actual=pred_cf,
                           target=graph.targets,
                           loss=0.0,
                           best_loss=self.best_loss,
                           x=graph.x,
                           edge_index=ego_edge_index,
                           edge_attr=None,
                           graph=graph,
                           oracle=oracle,
                           output_actual=out,
                           device=self.device)
        
        
        return self.finalize_explanation(context)
      

    @property
    def name(self):
        
        return "EGO" 