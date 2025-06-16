"""
Graph Perturber for COMBINEX Explainer.

This module provides the GraphPerturber class which implements graph-level perturbations
for generating counterfactual explanations in graph neural networks.
"""

from __future__ import annotations

import inspect
import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch_geometric.data import Data

from ....pertuber import Perturber
from ....utils.utils import discretize_to_nearest_integer
from src.datasets.datainfo import DataInfo


class GraphPerturber(Perturber):
    """
    Graph-level perturber for generating counterfactual explanations.
    
    This class implements perturbations for node features, edge weights, and edge attributes
    to generate counterfactual explanations for graph neural networks.
    
    Args:
        cfg: Configuration object containing model parameters.
        model: The graph neural network model to explain.
        graph: The input graph data.
        datainfo: Dataset information containing feature ranges and masks.
        device: Device to run computations on.
    """
    
    def __init__(
        self, 
        cfg: DictConfig, 
        model: nn.Module, 
        graph: Data,
        datainfo: DataInfo,
        device: str = "cuda"
    ) -> None:
        super().__init__(cfg=cfg, model=model)
        
        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        )
        
        # Hyperparameters
        self.beta = 0.5
        
        # Graph data
        self.graph_sample = graph
        self.edge_index = graph.edge_index
        self.x = graph.x
        
        # Dataset characteristics
        self.num_classes = datainfo.num_classes
        self.num_nodes = graph.x.shape[0]
        self.num_features = datainfo.num_features
        
        # Capability flags
        self.has_features = hasattr(datainfo, "num_features")
        self.has_edge_attrs = hasattr(datainfo, "discrete_edge_attr_mask")
        
        # Model capability inspection
        self._inspect_model_capabilities()
        
        # Initialize perturbation parameters
        self._initialize_perturbation_parameters(datainfo, graph)
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _inspect_model_capabilities(self) -> None:
        """Inspect model forward signature to determine supported arguments."""
        forward_signature = inspect.signature(self.model.forward)
        self.model_has_edge_weights = "edge_weights" in forward_signature.parameters
        self.model_has_edge_attr = "edge_attr" in forward_signature.parameters
        
    def _initialize_perturbation_parameters(self, datainfo: DataInfo, graph: Data) -> None:
        """Initialize all perturbation parameters based on data characteristics."""
        # Edge perturbation parameters
        self.EP_x = Parameter(torch.ones(len(graph.edge_index[0]), device=self.device))
        
        # Node feature perturbation setup
        if self.has_features:
            self._setup_feature_perturbations(datainfo)
            
        # Edge attribute perturbation setup
        if self.has_edge_attrs:
            self._setup_edge_attribute_perturbations(datainfo, graph)
            
    def _get_node_to_block(self, nodes_list: list[int] | None = None) -> Tensor:
        
        if not nodes_list:
            
            return torch.ones_like(self.x).long()
        
        else:
            
            mask = torch.ones_like(self.x)
            mask[nodes_list] = 0
            return mask.long()
        
    def _get_edge_attr_to_block(self, graph, edges_list: list[int] | None = None) -> Tensor:
        
        if not edges_list:
            
            return torch.ones_like(graph.edge_attr).long()
        
        else:
            
            mask = torch.ones_like(graph.edge_attr)
            mask[edges_list] = 0
            return mask.long()
            
            
    def _setup_feature_perturbations(self, datainfo: DataInfo) -> None:
        """Setup node feature perturbation parameters."""
        self.discrete_features_addition = True
        self.discrete_features_mask = datainfo.discrete_mask.to(self.device)
        self.continuous_features_mask = 1 - datainfo.discrete_mask.to(self.device)
        self.min_range = datainfo.min_range.to(self.device)
        self.max_range = datainfo.max_range.to(self.device)
        self.perturbation_mask = self._get_node_to_block()
        
        # Feature perturbation parameters
        self.P_x = Parameter(torch.zeros(
            self.num_nodes, self.num_features, device=self.device
        ))
        
    def _setup_edge_attribute_perturbations(self, datainfo: DataInfo, graph: Data) -> None:
        """Setup edge attribute perturbation parameters."""
        self.discrete_edge_attr_mask = datainfo.discrete_edge_attr_mask.to(self.device)
        self.edge_perturbation_mask = self._get_edge_attr_to_block()
        self.continuous_edge_attr_mask = 1 - datainfo.discrete_edge_attr_mask.to(self.device)
        self.min_range_edges = datainfo.min_range_edges.to(self.device)
        self.max_range_edges = datainfo.max_range_edges.to(self.device)
        
        # Edge attribute perturbation parameters
        self.E_attr = Parameter(torch.zeros(
            len(graph.edge_attr[0]), device=self.device
        ))
        
    @staticmethod
    def discretize_tensor(tensor: Tensor) -> Tensor:
        """
        Discretize tensor values to binary (0 or 1).
        
        Values <= 0.5 are mapped to 0, values > 0.5 are mapped to 1.
        
        Args:
            tensor: Input tensor to discretize.
            
        Returns:
            Discretized tensor with values in {0, 1}.
        """
        return torch.where(tensor <= 0.5, 0.0, 1.0)
    
    def _compute_perturbed_features(self, V_x: Tensor) -> Tensor:
        """
        Compute perturbed node features.
        
        Args:
            V_x: Original node features.
            
        Returns:
            Perturbed node features.
        """
        
        tanh_P = torch.tanh(self.P_x)
        scaled_P = self.min_range + (self.max_range - self.min_range) * tanh_P
        
        # Apply masked perturbations
        raw_features = (
            self.discrete_features_mask * (scaled_P * self.perturbation_mask + V_x) +
            self.continuous_features_mask * (self.P_x * self.perturbation_mask + V_x)
        )

        return torch.clamp(raw_features, min=self.min_range, max=self.max_range)
    
    def _compute_perturbed_edge_attributes(self) -> Optional[Tensor]:
        """
        Compute perturbed edge attributes if available.
        
        Returns:
            Perturbed edge attributes or None if not available.
        """
        if not self.model_has_edge_attr or not self.has_edge_attrs:
            return None
            
        # Scaled discrete base
        scaled_discrete = (
            self.min_range_edges + 
            (self.max_range_edges - self.min_range_edges) * torch.tanh(self.E_attr)
        )
        
        # Combine masked perturbations
        raw_edge_attr = (
            self.discrete_edge_attr_mask * (scaled_discrete * self.edge_perturbation_mask + self.graph_sample.edge_attr) +
            self.continuous_edge_attr_mask * (self.E_attr * self.edge_perturbation_mask + self.graph_sample.edge_attr)
        )
        
        return torch.clamp(raw_edge_attr, min=self.min_range_edges, max=self.max_range_edges)
    
    def _prepare_model_arguments(
        self, 
        perturbed_features: Tensor, 
        batch: Tensor,
        edge_weights: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        predict_edge_index: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """
        Prepare arguments for model forward pass.
        
        Args:
            perturbed_features: Perturbed node features.
            batch: Batch indices.
            edge_weights: Optional edge weights.
            edge_attr: Optional edge attributes.
            
        Returns:
            Dictionary of model arguments.
        """
        arguments = {
            "x": perturbed_features,
            "edge_index": self.graph_sample.edge_index,
            
        }
        
        if self.cfg.task.name.lower() == "graph":
            arguments["batch"] = batch
            
        if self.cfg.task.name.lower() == "link":
            arguments["predict_edge_index"] = predict_edge_index 
            
        if self.model_has_edge_weights and edge_weights is not None:
            arguments["edge_weights"] = edge_weights
            
        if self.model_has_edge_attr and edge_attr is not None:
            arguments["edge_attr"] = edge_attr
            
        return arguments
            
    def forward(self, V_x: Tensor, batch: Tensor = None, edge_to_predict = None) -> Tensor:
        """
        Forward pass with continuous perturbations.
        
        Args:
            V_x: Input node features.
            batch: Batch indices for nodes.
            
        Returns:
            Model output after applying perturbations.
        """
        # Compute perturbed features
        perturbed_features = self._compute_perturbed_features(V_x)
        
        # Compute perturbed edge attributes
        perturbed_edge_attr = self._compute_perturbed_edge_attributes()
        
        # Prepare model arguments
        edge_weights = torch.sigmoid(self.EP_x) if self.model_has_edge_weights else None
        arguments = self._prepare_model_arguments(
            perturbed_features, batch, edge_weights, perturbed_edge_attr, edge_to_predict
        )
        
        return self.model(**arguments)
    
    def _compute_discrete_feature_perturbations(self, V_x: Tensor) -> Tensor:
        """Compute discrete feature perturbations for prediction."""
        
        discrete_base = self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x)
        discrete_perturbation = self.discrete_features_mask * discretize_to_nearest_integer(
            discrete_base * self.perturbation_mask + V_x
        )
        discrete_perturbation = torch.clamp(
            discrete_perturbation, min=self.min_range, max=self.max_range
        )
        
        continuous_perturbation = self.continuous_features_mask * torch.clamp(
            self.P_x * self.perturbation_mask + V_x, min=self.min_range, max=self.max_range
        )
        
        features = discrete_perturbation + continuous_perturbation
        
        
        return features
    
    def _compute_discrete_edge_attribute_perturbations(self) -> Optional[Tensor]:
        """Compute discrete edge attribute perturbations for prediction."""
        if not self.model_has_edge_attr or not self.has_edge_attrs:
            return None
            
        # Discrete edge perturbations
        edge_base = (
            self.min_range_edges + 
            (self.max_range_edges - self.min_range_edges) * torch.tanh(self.E_attr)
        )
        edge_discrete = self.discrete_edge_attr_mask * discretize_to_nearest_integer(
            edge_base * self.edge_perturbation_mask + self.graph_sample.edge_attr
        )
        edge_discrete = torch.clamp(
            edge_discrete, min=self.min_range_edges, max=self.max_range_edges
        )
        
        # Continuous edge perturbations
        edge_continuous = self.continuous_edge_attr_mask * torch.clamp(
            self.E_attr * self.edge_perturbation_mask + self.graph_sample.edge_attr,
            min=self.min_range_edges, 
            max=self.max_range_edges
        )
        
        return edge_discrete + edge_continuous
    
    def forward_prediction(self, V_x: Tensor, batch: Tensor = None, edge_to_predict = None) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Forward pass with discrete perturbations for final prediction.
        
        Args:
            V_x: Input node features.
            batch: Batch indices for nodes.
            
        Returns:
            Tuple of (model_output, perturbed_features, edge_weights, perturbed_edge_attr).
        """
        # Compute discrete perturbations
        V_pert = self._compute_discrete_feature_perturbations(V_x)
        EP_x_discrete = self.discretize_tensor(torch.sigmoid(self.EP_x))
        perturbed_edge_attr = self._compute_discrete_edge_attribute_perturbations()
        
        # Prepare model arguments
        arguments = self._prepare_model_arguments(
            V_pert, batch, EP_x_discrete, perturbed_edge_attr, edge_to_predict
        )
        
        output = self.model(**arguments)
        
        return output, V_pert, self.EP_x, perturbed_edge_attr
    
    def edge_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:
        """
        Calculate edge-based loss for sparsity regularization.
        
        Args:
            graph: Input graph data.
            
        Returns:
            Tuple of (sparsity_loss, counterfactual_edge_index).
        """
        # Generate perturbed edge weights
        cf_edge_weights = torch.sigmoid(self.EP_x)
        
        # Sparsity loss: penalize deviations from original weights
        sparsity_loss = torch.sum(torch.abs(cf_edge_weights - 1.0))
        
        # Generate discrete edge index
        cf_edge_weights_discrete = self.discretize_tensor(cf_edge_weights)
        cf_edge_index = graph.edge_index[:, cf_edge_weights_discrete == 1]
        
        return sparsity_loss, cf_edge_index
    
    def node_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:
        """
        Calculate node feature perturbation loss.
        
        Args:
            graph: Input graph data.
            
        Returns:
            Tuple of (total_loss, edge_index).
        """
        # Discrete feature loss (L1)
        discrete_target = torch.clamp(
            self.discrete_features_mask * F.tanh(self.P_x) + graph.x,
            self.min_range, 
            self.max_range
        )
        loss_discrete = F.l1_loss( 
            discrete_target, graph.x * self.discrete_features_mask
        )
        
        # Continuous feature loss (MSE)
        continuous_target = self.continuous_features_mask * (self.P_x + graph.x)
        loss_continuous = F.mse_loss(
            graph.x * self.continuous_features_mask, 
            continuous_target
        )
        
        total_loss = loss_discrete + loss_continuous
        
        return total_loss, self.edge_index
    
    def edge_attr_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:
        """
        Calculate edge attribute perturbation loss.
        
        Args:
            graph: Input graph data.
            
        Returns:
            Tuple of (total_loss, edge_index).
        """
        # Discrete edge attribute loss (L1)
        discrete_target = torch.clamp(
            self.discrete_edge_attr_mask * F.tanh(self.E_attr) + graph.edge_attr,
            self.min_range_edges,
            self.max_range_edges
        )
        loss_discrete = F.l1_loss(
            graph.edge_attr * self.discrete_edge_attr_mask,
            discrete_target
        )
        
        # Continuous edge attribute loss (MSE)
        continuous_target = self.continuous_edge_attr_mask * (self.E_attr + graph.edge_attr)
        loss_continuous = F.mse_loss(
            graph.edge_attr * self.continuous_edge_attr_mask,
            continuous_target
        )
        
        total_loss = loss_discrete + loss_continuous
        
        return total_loss, self.edge_index
    
    def get_perturbation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current perturbation parameters.
        
        Returns:
            Dictionary containing perturbation statistics.
        """
        summary = {
            "num_nodes": self.num_nodes,
            "num_features": self.num_features,
            "num_edges": len(self.edge_index[0]),
            "has_features": self.has_features,
            "has_edge_attrs": self.has_edge_attrs,
            "model_has_edge_weights": self.model_has_edge_weights,
            "model_has_edge_attr": self.model_has_edge_attr
        }
        
        if self.has_features:
            summary.update({
                "feature_perturbation_norm": torch.norm(self.P_x).item(),
                "feature_perturbation_max": torch.max(torch.abs(self.P_x)).item()
            })
            
        if self.has_edge_attrs:
            summary.update({
                "edge_attr_perturbation_norm": torch.norm(self.E_attr).item(),
                "edge_attr_perturbation_max": torch.max(torch.abs(self.E_attr)).item()
            })
            
        summary.update({
            "edge_weight_perturbation_norm": torch.norm(self.EP_x - 1.0).item(),
            "edge_weight_min": torch.min(torch.sigmoid(self.EP_x)).item(),
            "edge_weight_max": torch.max(torch.sigmoid(self.EP_x)).item()
        })
        
        return summary