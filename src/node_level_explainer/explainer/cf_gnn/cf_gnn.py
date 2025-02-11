# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from src.abstract.explainer import Explainer
from ...utils.utils import build_counterfactual_graph, get_optimizer
from omegaconf import DictConfig
from torch_geometric.data import Data
import time
from .edge_perturber import EdgePerturber   
from torch import nn




class CFExplainer(Explainer):

    def __init__(self, cfg: DictConfig, datainfo):
        super().__init__(cfg=cfg, datainfo=datainfo)   
        
        self.set_reproducibility()
        
    def explain(self, graph: Data, oracle: nn.Module) -> Data:
        """
        Explain a node in the given graph using the oracle model.

        This method freezes all parameters in the oracle model and initializes an 
        EdgePerturber with the given configuration, number of classes, oracle model, 
        and graph. It then sets up an optimizer for the EdgePerturber and iteratively 
        trains it to find the best counterfactual example for the node.

        Args:
            graph (Data): The input graph data.
            oracle (nn.Module): The oracle model used for node classification.

        Returns:
            Data: The best counterfactual example found during training.
        """
        

        self.edge_perturber = EdgePerturber(cfg=self.cfg, 
                                            num_classes=self.datainfo.num_classes, 
                                            model=oracle, 
                                            graph=graph)
        
        # Freeze the oracle model parameters
        self.edge_perturber.deactivate_model()

        # Get the optimizer for the edge perturber
        self.optimizer = get_optimizer(self.cfg, self.edge_perturber)
        
        self.best_loss = np.inf
        best_cf_example = None
        graph = graph.to(self.device)
        start_time = time.time()

        for _ in range(self.cfg.optimizer.num_epochs):
            new_sample = self.train(graph, oracle)

            if time.time() - start_time > self.cfg.timeout:
                break

            if new_sample is not None:
                best_cf_example = new_sample

        return best_cf_example


    def train(self, graph: Data, oracle: nn.Module) -> Data:
        """
        Trains the edge perturber model on the given graph data.

        Args:
            graph (Data): The input graph data containing node features, edge index, and other attributes.
            oracle (nn.Module): The oracle model used for predictions.

        Returns:
            Data: The counterfactual graph if the conditions are met, otherwise None.

        The training process involves:
        - Zeroing the gradients of the optimizer.
        - Performing a forward pass through the edge perturber model.
        - Calculating the non-differentiable target using the oracle's predictions.
        - Computing the loss and performing backpropagation.
        - Clipping the gradients to prevent exploding gradients.
        - Updating the model parameters using the optimizer.
        - Building a counterfactual graph if the conditions on fidelity loss and sparsity loss are satisfied.
        """
        
        # Zero the gradients of the optimizer
        self.optimizer.zero_grad()
        
        # Perform a forward pass through the edge perturber model
        output = self.edge_perturber.forward()    
        output_actual = self.edge_perturber.forward_prediction()
        
        # Calculate the non-differentiable target using the oracle's predictions
        y_non_differentiable = torch.argmax(output_actual[graph.new_idx])
        
        # Compute the loss
        loss, edge_index = self.edge_perturber.loss(graph, output, y_non_differentiable)
        
        # Perform backpropagation
        loss.backward()
        
        # Clip the gradients to prevent exploding gradients
        clip_grad_norm_(self.edge_perturber.parameters(), 2.0)
        
        # Update the model parameters using the optimizer
        self.optimizer.step()

        counterfactual = None

        # Build a counterfactual graph if the conditions on fidelity loss and sparsity loss are satisfied
        if y_non_differentiable == graph.targets[graph.new_idx] and loss.item() < self.best_loss:
            
            counterfactual = build_counterfactual_graph(graph.x, edge_index, graph, oracle, output_actual)
            
            self.best_loss = loss.item()

        return counterfactual


    @property
    def name(self):

        return "CF-GNNExplainer"
    