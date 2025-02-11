from omegaconf import DictConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Tuple
from ...utils.utils import discretize_to_nearest_integer
from ...perturber.pertuber import Perturber   
from src.datasets.dataset import DataInfo


class NodePerturber(Perturber):

    def __init__(self, 
                    cfg: DictConfig, 
                    model: nn.Module, 
                    graph: Data,
                    datainfo: DataInfo,
                    device: str = "cuda") -> None:
        
        super().__init__(cfg=cfg, model=model)
        
        
        self.device = device
        
        # Dataset characteristics
        self.num_classes = datainfo.num_classes
        self.num_nodes = graph.x.shape[0]
        self.num_features = datainfo.num_features
        self.min_range = datainfo.min_range.to(device)
        self.max_range = datainfo.max_range.to(device)
        
        # Explainer characteristics
        self.discrete_features_addition: bool = True
        self.discrete_features_mask: Tensor = datainfo.discrete_mask.to(device)
        self.continous_features_mask: Tensor = 1 - datainfo.discrete_mask.to(device)
        # Model's parameters
        self.P_x = Parameter(torch.zeros(self.num_nodes, self.num_features, device=self.device))
        
        # Graph's components
        self.edge_index = graph.edge_index
        self.x = graph.x
    
    def forward(self, V_x: Tensor, batch) -> Tensor:
        """
        Forward pass for the NodePerturber.

        Args:
        V_x (Tensor): The input feature tensor.
        adj (Tensor): The adjacency matrix.

        Returns:
        Tensor: The output of the model after applying perturbations.
        """
        
        
        discrete_perturbation = self.discrete_features_mask * torch.clamp(self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x) + V_x, min=self.min_range, max=self.max_range)
        
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        
        V_pert = discrete_perturbation + continuous_perturbation

        return self.model(V_pert, self.edge_index, batch)

    
    def forward_prediction(self, V_x: Tensor, batch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward prediction for the NodePerturber.

        Args:
        V_x (Tensor): The input feature tensor.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: The output of the model, the perturbed features, and the perturbation tensor.
        """
        discrete_perturbation = self.discrete_features_mask * discretize_to_nearest_integer(self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x) + V_x)
        
        discrete_perturbation = torch.clamp(discrete_perturbation, min=self.min_range, max=self.max_range)
        
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        
        V_pert = discrete_perturbation + continuous_perturbation

        out = self.model(V_pert, self.edge_index, batch)
        return out, V_pert, self.P_x
    
    
    def loss(self, graph: Data, output: Tensor, y_node_non_differentiable) -> Tuple[Tensor, dict, Tensor]:
        """
        Computes the loss for the NodePerturber.

        Args:
        graph: The input graph.
        output: The model output.
        y_node_non_differentiable: The non-differentiable node labels.

        Returns:
        Tuple containing the total loss, a dictionary of individual losses, and the adjacency matrix.
        """
        
        
        y_node_predicted = output
        y_target = graph.targets.unsqueeze(0)
        
        constant = ((y_target != torch.argmax(y_node_predicted)) or (y_target != y_node_non_differentiable)).float()
        
        loss_pred = F.cross_entropy(y_node_predicted, y_target)
        loss_discrete = F.l1_loss(graph.x * self.discrete_features_mask, torch.clamp(self.discrete_features_mask * F.tanh(self.P_x) + graph.x, 0, 1))
        loss_continue = F.mse_loss(graph.x * self.continous_features_mask, self.continous_features_mask * (self.P_x + graph.x))
        
        loss_total = constant * loss_pred + loss_discrete + loss_continue
        
        return loss_total, self.edge_index
