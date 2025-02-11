import time
from torch_geometric.data import Data
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from .node_perturber import NodePerturber
from omegaconf import DictConfig
from ...utils.utils import build_counterfactual_graph, get_optimizer
from ....abstract.explainer import Explainer  
from tqdm import tqdm 
from src.datasets.dataset import DataInfo


class CFExplainerFeatures(Explainer):
    """
    CF Explainer class, returns counterfactual subgraph
    """
    def __init__(self, cfg: DictConfig, datainfo: DataInfo):
        super().__init__(cfg=cfg, datainfo=datainfo)
        self.discrete_mask = None
        self.set_reproducibility()


    def explain(self, graph: Data, oracle):

        self.best_loss = np.inf
        self.node_perturber = NodePerturber(cfg=self.cfg, 
                                      model=oracle,
                                      datainfo=self.datainfo, 
                                      graph=graph).to(self.device)
        
        self.node_perturber.deactivate_model()
        
        self.optimizer = get_optimizer(self.cfg, self.node_perturber)
        best_cf_example = None
        
        start = time.time()

        for epoch in range(self.cfg.optimizer.num_epochs):
            
            new_sample = self.train(graph, oracle)
            
            if time.time() - start > self.cfg.timeout:
                
                return best_cf_example

            if new_sample is not None:
                best_cf_example = new_sample

        return best_cf_example
    

    def train(self, graph: Data, oracle) -> Data:

        self.optimizer.zero_grad()

        differentiable_output = self.node_perturber.forward(graph.x) 
        model_out, V_pert, P_x = self.node_perturber.forward_prediction(graph.x) 
        
        y_pred_new_actual = torch.argmax(model_out, dim=1)

        loss, edge_index = self.node_perturber.loss(graph, differentiable_output, y_pred_new_actual[graph.new_idx])
        loss.backward()

        clip_grad_norm_(self.node_perturber.parameters(), self.cfg.explainer.clip_grad_norm)
        self.optimizer.step()

        counterfactual = None

        if y_pred_new_actual[graph.new_idx] == graph.targets[graph.new_idx] and loss.item() < self.best_loss:
            
            counterfactual = build_counterfactual_graph(x=V_pert,
                                       edge_index=edge_index, 
                                       graph=graph, 
                                       oracle=oracle, 
                                       output_actual=model_out, 
                                       device=self.device)

            self.best_loss = loss.item()

        return counterfactual
    
    @property
    def name(self):
        """
        Property that returns the name of the explainer.
        Returns:
            str: The name of the explainer, "CF-GNNExplainer Features".
        """
        return "CF-GNNExplainer Features" 
    