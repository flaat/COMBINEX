import time
from torch_geometric.data import Data
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from .graph_perturber import GraphPerturber
from omegaconf import DictConfig
from ...utils.utils import build_counterfactual_graph, get_optimizer
from ....abstract.explainer import Explainer  
from tqdm import tqdm 
from src.datasets.dataset import DataInfo


class CombinedExplainer(Explainer):
    """
    CF Explainer class, returns counterfactual subgraph
    """
    def __init__(self, cfg: DictConfig, datainfo: DataInfo):
        super().__init__(cfg=cfg, datainfo=datainfo)
        self.discrete_mask = None
        self.set_reproducibility()


    def explain(self, graph: Data, oracle):

        self.best_loss = np.inf
        self.graph_perturber = GraphPerturber(cfg=self.cfg, 
                                      model=oracle,
                                      datainfo=self.datainfo, 
                                      graph=graph,
                                      device="cuda").to(self.device)
        
        self.graph_perturber.deactivate_model()
        
        self.optimizer = get_optimizer(self.cfg, self.graph_perturber)
        best_cf_example = None
        
        start = time.time()

        for epoch in range(self.cfg.optimizer.num_epochs):
            
            new_sample = self.train(graph, oracle, epoch)
            
            if time.time() - start > self.cfg.timeout:
                
                return best_cf_example

            if new_sample is not None:
                best_cf_example = new_sample

        return best_cf_example
    

    def train(self, graph: Data, oracle, epoch) -> Data:
        
        

        self.optimizer.zero_grad()

        differentiable_output = self.graph_perturber.forward(graph.x) 
        model_out, _, V_pert, EP_x = self.graph_perturber.forward_prediction(graph.x) 
        
        y_pred_new_actual = torch.argmax(model_out, dim=1)
        
        edge_loss, cf_edges = self.graph_perturber.edge_loss(graph)
        node_loss, _ = self.graph_perturber.node_loss(graph)
        
        alpha = self.get_alpha(epoch, edge_loss, node_loss)

        node_to_explain = graph.new_idx

        # Model's prediction for the node to explain
        y_node_predicted = differentiable_output[node_to_explain].unsqueeze(0)
        y_node_predicted_non_diff = model_out[node_to_explain].unsqueeze(0) # Original ground truth

        y_target = graph.targets[node_to_explain].unsqueeze(0)  # Counterfactual target
                
        eta = ((y_target != torch.argmax(y_node_predicted)) or (y_target != torch.argmax(y_node_predicted_non_diff))).float()
        
        loss_pred = torch.nn.functional.cross_entropy(y_node_predicted, y_target)        
        
        loss = eta * loss_pred + (1 - alpha) * edge_loss + alpha * node_loss
        
        loss.backward()
        self.optimizer.step()

        counterfactual = None

        if y_pred_new_actual[graph.new_idx] == graph.targets[graph.new_idx] and loss.item() < self.best_loss:
            
            counterfactual = build_counterfactual_graph(x=V_pert,
                                       edge_index=cf_edges, 
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
        return "CombinedExplainer" 
    
    
    
    def get_alpha(self, epoch: int, edge_loss, node_loss) -> float:
        """
        Scheduler for the alpha value that implements different policies.
        Args:
            epoch (int): The current epoch number.
        Returns:
            float: The scheduled alpha value.
        """
        if self.cfg.scheduler.policy == "linear":
            # Linear decay
            alpha = max(0.0, 1.0 - epoch / self.cfg.optimizer.num_epochs)
        elif self.cfg.scheduler.policy == "exponential":
            # Exponential decay
            alpha = max(0.0, np.exp(-epoch / self.cfg.scheduler.decay_rate))
        elif self.cfg.scheduler.policy == "sinusoidal":
            # Sinusoidal decay
            alpha = max(0.0, 0.5 * (1 + np.cos(np.pi * epoch / self.cfg.optimizer.num_epochs)))
        elif self.cfg.scheduler.policy == "dynamic":
            # Dynamic adjustment based on loss values
            alpha = 0.0 if edge_loss > node_loss else 1.0
        else:
            # Default to a constant alpha
            alpha = self.cfg.scheduler.initial_alpha
        return alpha
