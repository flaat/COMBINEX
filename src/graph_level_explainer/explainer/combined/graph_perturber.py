from omegaconf import DictConfig
import logging
import torch
from ...perturber.pertuber import Perturber
from torch_geometric.data import Data   
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Tuple
from ...utils.utils import discretize_to_nearest_integer
from src.datasets.datainfo import DataInfo
import inspect

class GraphPerturber(Perturber):

    def __init__(self, 
                    cfg: DictConfig, 
                    model: nn.Module, 
                    graph: Data,
                    datainfo: DataInfo,
                    device: str = "cuda") -> None:
        
        super().__init__(cfg=cfg, model=model)
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
        self.beta = 0.5
        
        
        self.EP_x = Parameter(torch.ones(len(graph.edge_index[0]), device=self.device))
        self.graph_sample = graph
        self.device = device
        
        # Dataset characteristics
        self.num_classes = datainfo.num_classes
        self.num_nodes = graph.x.shape[0]
        self.num_features = datainfo.num_features

        
        self.has_features = hasattr(datainfo, "num_features")
        self.has_edge_attrs = hasattr(datainfo, "discrete_edge_attr_mask")
        
        self.model_has_edge_weights = inspect.signature(self.model.forward).parameters.get("edge_weights") is not None
        self.model_has_edge_attr = inspect.signature(self.model.forward).parameters.get("edge_attr") is not None
        
        if self.has_features:
            # Masks characteristics
            self.discrete_features_addition: bool = True
            self.discrete_features_mask: Tensor = datainfo.discrete_mask.to(device)
            self.continous_features_mask: Tensor = 1 - datainfo.discrete_mask.to(device)
            self.min_range = datainfo.min_range.to(device)
            self.max_range = datainfo.max_range.to(device)        
            # Features perturbation
            self.P_x = Parameter(torch.zeros(self.num_nodes, self.num_features, device=self.device))        
        
        if self.has_edge_attrs:
            
            self.discrete_edge_attr_mask: Tensor = datainfo.discrete_edge_attr_mask.to(device)
            self.continous_edge_attr_mask: Tensor = 1 - datainfo.discrete_edge_attr_mask.to(device) 
            self.min_range_edges = datainfo.min_range_edges.to(device)
            self.max_range_edges = datainfo.max_range_edges.to(device)
   
            self.E_attr = Parameter(torch.zeros(len(graph.edge_attr[0]), device=self.device))            
        
        
        # Graph's components
        self.edge_index = graph.edge_index
        self.x = graph.x         
        
        
    def discretize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Discretizes the input tensor based on the following rules:
        - Values less than or equal to -0.5 are set to -1
        - Values equal to 0.5 are set to 1
        - All other values are set to 0

        Args:
        tensor (torch.Tensor): The input tensor to be discretized.

        Returns:
        torch.Tensor: The discretized tensor.
        """
        discretized_tensor = torch.where(tensor <= 0.5, 0, 1)
        return discretized_tensor.float()
    
            
    def forward(self, V_x, batch) -> Tensor:       
        r"""
        Forward pass for the graph perturber.
        This method perturbs the input features and passes them through the model.
        
        Args:
            V_x (Tensor): The input node features.
            batch (Tensor): The batch indices for the nodes.
        Returns:
            Tensor: The output of the model after applying the perturbations.
        """
        
        perturbed_edge_attr = None
        tanh_P  = torch.tanh(self.P_x)
        scaled_P = self.min_range + (self.max_range - self.min_range) * tanh_P

        # build masked perturbations and clamp once
        raw = (
            self.discrete_features_mask   * (scaled_P         + V_x) +
            self.continous_features_mask  * (self.P_x         + V_x)
        )

        perturbed_features = torch.clamp(raw, min=self.min_range, max=self.max_range)
        
        arguments = {"x": perturbed_features, "edge_index": self.graph_sample.edge_index, "batch": batch}
        
        if self.model_has_edge_weights:
            arguments["edge_weights"] = torch.sigmoid(self.EP_x)
            
        if self.model_has_edge_attr:
            
            # 1) build the “scaled” discrete base
            scaled_discrete = (self.min_range_edges + (self.max_range_edges - self.min_range_edges) * torch.tanh(self.E_attr))

            # 2) combine the two masked perturbations
            raw = (self.discrete_edge_attr_mask * (scaled_discrete + self.graph_sample.edge_attr)
                + self.continous_edge_attr_mask * (self.E_attr + self.graph_sample.edge_attr))

            # 3) single clamp to enforce your min/max
            perturbed_edge_attr = torch.clamp(raw, min=self.min_range_edges, max=self.max_range_edges)
            arguments["edge_attr"] = perturbed_edge_attr
    
        return self.model(**arguments)

    
    def forward_prediction(self, V_x, batch):
        
        perturbed_edge_attr = None
        
        discrete_perturbation = self.discrete_features_mask * discretize_to_nearest_integer(self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x) + V_x)
        discrete_perturbation = torch.clamp(discrete_perturbation, min=self.min_range, max=self.max_range)
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        V_pert = discrete_perturbation + continuous_perturbation
        EP_x_discrete = self.discretize_tensor(torch.sigmoid(self.EP_x))
        
        arguments = {"x": V_pert, "edge_index": self.graph_sample.edge_index, "batch": batch}
        
        if self.model_has_edge_weights:
            arguments["edge_weights"] = EP_x_discrete
            
        if self.model_has_edge_attr:
            
            edge_perturbation_discrete_rescaling = self.discrete_edge_attr_mask * discretize_to_nearest_integer(self.min_range_edges + (self.max_range_edges - self.min_range_edges) * torch.tanh(self.E_attr) + self.graph_sample.edge_attr)
            edge_perturbation_discrete = torch.clamp(edge_perturbation_discrete_rescaling, min=self.min_range_edges, max=self.max_range_edges)
            edge_perturbation_continuous = self.continous_edge_attr_mask * torch.clamp((self.E_attr + self.graph_sample.edge_attr), min=self.min_range_edges, max=self.max_range_edges)
            perturbed_edge_attr = edge_perturbation_discrete + edge_perturbation_continuous        
            arguments["edge_attr"] = perturbed_edge_attr
        
        out = self.model(**arguments)     
                     
        return out, V_pert, self.EP_x, perturbed_edge_attr
    
    
    def edge_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:

        # Generate perturbed edge index (with edge weights)
        cf_edge_weights = torch.sigmoid(self.EP_x)  # Learnable edge weights (perturbations)

        # Graph sparsity loss: Penalize large changes in edge weights
        sparsity_loss = torch.sum(torch.abs(cf_edge_weights - 1))  # Penalize deviations from original weights
        cf_edge_weights_discrete = self.discretize_tensor(cf_edge_weights)
        cf_edge_index = graph.edge_index[:, cf_edge_weights_discrete == 1]
        
        return sparsity_loss, cf_edge_index

    def node_loss(self, graph: Data) -> Tuple[Tensor, dict, Tensor]:
        """
        Calculate the node loss for a given graph.
        This method computes the loss for discrete and continuous features of the nodes in the graph.
        The discrete feature loss is calculated using L1 loss, and the continuous feature loss is 
        calculated using Mean Squared Error (MSE) loss. The total loss is the sum of these two losses.
        Args:
            graph (Data): The input graph data containing node features.
        Returns:
            Tuple[Tensor, dict, Tensor]: A tuple containing the total loss, edge index, and any additional 
            information (if applicable).
        """

        loss_discrete = F.l1_loss(graph.x * self.discrete_features_mask, torch.clamp(self.discrete_features_mask * F.tanh(self.P_x) + graph.x, self.min_range, self.max_range))
        loss_continue = F.mse_loss(graph.x * self.continous_features_mask, self.continous_features_mask * (self.P_x + graph.x))
        loss_total = loss_discrete + loss_continue
        
        return loss_total, self.edge_index
    
    def edge_attr_loss(self, graph: Data) -> Tuple[Tensor, dict, Tensor]:
        """
        Calculate the node loss for a given graph.
        This method computes the loss for discrete and continuous features of the nodes in the graph.
        The discrete feature loss is calculated using L1 loss, and the continuous feature loss is 
        calculated using Mean Squared Error (MSE) loss. The total loss is the sum of these two losses.
        Args:
            graph (Data): The input graph data containing node features.
        Returns:
            Tuple[Tensor, dict, Tensor]: A tuple containing the total loss, edge index, and any additional 
            information (if applicable).
        """

        loss_discrete = F.l1_loss(graph.edge_attr * self.discrete_edge_attr_mask, torch.clamp(self.discrete_edge_attr_mask * F.tanh(self.E_attr) + graph.edge_attr, self.min_range_edges, self.max_range_edges))
        loss_continue = F.mse_loss(graph.edge_attr * self.continous_edge_attr_mask, self.continous_edge_attr_mask * (self.E_attr + graph.edge_attr))
        loss_total = loss_discrete + loss_continue
        
        return loss_total, self.edge_index