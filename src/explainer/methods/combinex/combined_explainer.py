import time
from torch_geometric.data import Data
import torch
import numpy as np
from .graph_perturber import GraphPerturber
from omegaconf import DictConfig
from ....utils.utils import (
    get_optimizer, 
    ExplanationContext
    )
from ....abstract.explainer import ExplainerABC  
from tqdm import tqdm 
from src.datasets.datainfo import DataInfo


class Combinex(ExplainerABC):
    """
    CF ExplainerABC class, returns counterfactual subgraph
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
        """
        Trains the graph perturber for one epoch and returns a counterfactual example if found.
        
        Args:
            graph (Data): The input graph data.
            oracle: The oracle model used for predictions.
            epoch (int): The current epoch number.
        
        Returns:
            Data: The counterfactual example if found, otherwise None.
        """

        
        self.optimizer.zero_grad()
        args = None
        
        if self.cfg.task.name.lower() == "node":
            
            args = (graph.x,)
        elif self.cfg.task.name.lower() == "graph":
            args = (graph.x, graph.batch)
        elif self.cfg.task.name.lower() == "link":
            args = (graph.x, graph.edge_index, graph.edge_index_predict)
            
        differentiable_output = self.graph_perturber.forward(*args) 
        model_out, V_pert, EP_x, edge_attr_pert = self.graph_perturber.forward_prediction(*args) 
        
        if self.cfg.task.name.lower() != "link":
        
            y_pred_new_actual = torch.argmax(model_out, dim=1)
            y_pred_differentiable = torch.argmax(differentiable_output, dim=1)
            
        else:
            y_pred_new_actual = torch.where(model_out>0.5, 1, 0)
            y_pred_differentiable = differentiable_output
        
        edge_loss, cf_edges = self.graph_perturber.edge_loss(graph)
        node_loss, _ = self.graph_perturber.node_loss(graph)
        
        
        alpha = self.get_alpha(epoch, edge_loss, node_loss)
        
        target = None
        
        if self.cfg.task.name.lower() == "graph" or self.cfg.task.name.lower() == "link":
            eta = ((y_pred_new_actual != graph.targets) or (graph.targets != y_pred_differentiable)).float()
            target = graph.targets.unsqueeze(0)
            
        elif self.cfg.task.name.lower() == "node":
            
            eta = ((y_pred_new_actual[graph.new_idx] != graph.targets[graph.new_idx]) or (graph.targets[graph.new_idx] != y_pred_differentiable[graph.new_idx])).float()
            target = graph.targets

        if self.cfg.task.name.lower() == "link":
            
            target = target.float()
        
        loss_pred = torch.nn.functional.cross_entropy(differentiable_output, target)            
        if self.graph_perturber.has_edge_attrs:
            edge_attr_loss,_ = self.graph_perturber.edge_attr_loss(graph)
            
            beta = self.cfg.explainer.beta if self.cfg.explainer.beta != "None" else ((1-alpha)/2)
            p_lambda = self.cfg.explainer.p_lambda if self.cfg.explainer.p_lambda != "None" else ((1-alpha)/2)
            loss = eta * loss_pred + alpha * edge_loss + beta * node_loss + p_lambda * edge_attr_loss
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item():.8f}, Pred Loss: {loss_pred} Edge Loss: {edge_loss.item():.4f}, Node Loss: {node_loss.item():.4f} Edge Attr. Loss:{edge_attr_loss.item():.8f}")
    
        else:
            loss = eta * loss_pred + alpha * edge_loss + (1-alpha) * node_loss
        
        

        loss.backward()        
        self.optimizer.step()
        
        context = ExplanationContext(y_pred_new_actual=y_pred_new_actual,
                           target=graph.targets,
                           loss=loss,
                           best_loss=self.best_loss,
                           x=V_pert,
                           edge_index=cf_edges,
                           edge_attr=edge_attr_pert,
                           graph=graph,
                           oracle=oracle,
                           output_actual=model_out,
                           device=self.device)
        
        
        return self.finalize_explanation(context)
    
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
            # Exponential decaygraph
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
