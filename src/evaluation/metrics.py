"""
Unified Metrics for COMBINEX Explainer.

This module provides a comprehensive set of metrics for evaluating both node-level 
and graph-level counterfactual explanations in graph neural networks.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Optional, Union, Tuple
import warnings

import torch
from torch_geometric.data import Data


class TaskType(Enum):
    """Enumeration for different explanation task types."""
    NODE = "node"
    GRAPH = "graph"


class MetricCalculator:
    """
    Unified metric calculator for both node-level and graph-level explanations.
    
    This class provides a comprehensive set of metrics for evaluating the quality
    of counterfactual explanations across different tasks.
    """
    
    def __init__(self, task_type: TaskType = TaskType.NODE):
        """
        Initialize the metric calculator.
        
        Args:
            task_type: The type of explanation task (node or graph level).
        """
        self.task_type = task_type
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def compute_all_metrics(
        self, 
        factual: Data, 
        counterfactual: Optional[Data] = None,
        mean_projection: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics for the given data.
        
        Args:
            factual: The factual data instance.
            counterfactual: The counterfactual data instance (optional).
            mean_projection: Mean projection vector for distance calculations (optional).
            
        Returns:
            Dictionary containing all computed metrics.
        """
        metrics = {}
        
        try:
            # Distance metrics (require mean projection)
                
            if counterfactual is not None:
                metrics.update(self._compute_counterfactual_distance_metrics(
                    factual, counterfactual, mean_projection
                ))
                metrics.update(self._compute_comparison_metrics(factual, counterfactual))      

                if mean_projection is not None:
                    metrics.update(self._compute_distance_metrics(factual, mean_projection))
            else:
                
                metrics.update({"validity": 0.0})   
        except Exception as e:
            self.logger.error(f"Error computing metrics: {e}")
            warnings.warn(f"Some metrics could not be computed: {e}")
            
        return metrics
    
    def _compute_distance_metrics(
        self, 
        sample: Data, 
        mean: torch.Tensor
    ) -> Dict[str, float]:
        """Compute distance-based metrics."""
        metrics = {}
        
        
        # Projection distance (if available)
        if hasattr(sample, 'x_projection') and sample.x_projection is not None:
            metrics["sample_distance_from_mean_projection"] = (
                sample_distance_from_mean_projection(mean, sample)
            )
            
        return metrics
    
    def _compute_counterfactual_distance_metrics(
        self,
        factual: Data,
        counterfactual: Data,
        mean: torch.Tensor
    ) -> Dict[str, float]:
        """Compute counterfactual distance metrics."""
        metrics = {}
        
        # Factual-counterfactual distance
        if (hasattr(factual, 'x_projection') and hasattr(counterfactual, 'x_projection') and
            factual.x_projection is not None and counterfactual.x_projection is not None):
            metrics["factual_counterfactual_distance"] = (
                factual_counterfactual_distance(factual, counterfactual)
            )
            
        return metrics
    
    def _compute_comparison_metrics(
        self, 
        factual: Data, 
        counterfactual: Data
    ) -> Dict[str, float]:
        """Compute metrics comparing factual and counterfactual data."""
        metrics = {"validity": 1}  # Initialize validity metric
        
        # Fidelity
        metrics["fidelity"] = fidelity(factual, counterfactual)
        
        # Sparsity metrics
        metrics["node_sparsity"] = node_sparsity(factual, counterfactual)
        metrics["edge_sparsity"] = edge_sparsity(factual, counterfactual)
        
        # Edge attribute sparsity (if available)
        if (hasattr(factual, 'edge_attr') and hasattr(counterfactual, 'edge_attr') and
            factual.edge_attr is not None and counterfactual.edge_attr is not None):
            metrics["edge_attr_sparsity"] = edge_attr_sparsity(factual, counterfactual)
        
        # Graph edit distance
        metrics["graph_edit_distance"] = graph_edit_distance(factual, counterfactual)
        
        # Perturbation distance
        metrics["perturbation_distance"] = perturbation_distance(factual, counterfactual)
                    
        return metrics


# =============================================================================
# DISTANCE METRICS
# =============================================================================

def sample_distance_from_mean(mean: torch.Tensor, sample: Data) -> float:
    """
    Calculate the Euclidean distance between the mean vector and the mean of the sample's features.

    Args:
        mean: The mean vector.
        sample: The sample data instance.

    Returns:
        The Euclidean distance.
    """
    if sample.x is None or sample.x.numel() == 0:
        return 0.0
        
    # Calculate the mean of the sample's features
    sample_mean = torch.mean(sample.x_projection, dim=0)
    
    # Ensure tensors are on the same device
    if mean.device != sample_mean.device:
        mean = mean.to(sample_mean.device)
    
    # Calculate the Euclidean distance
    distance_val = torch.sqrt(torch.sum((sample_mean - mean) ** 2))
    return distance_val.item()


def sample_distance_from_mean_projection(mean: torch.Tensor, sample: Data) -> float:
    """
    Calculate the Euclidean distance between the mean vector and the sample's projected features.

    Args:
        mean: The mean vector.
        sample: The sample data instance.

    Returns:
        The Euclidean distance.
    """
    if not hasattr(sample, 'x_projection') or sample.x_projection is None:
        warnings.warn("Sample does not have x_projection attribute")
        return 0.0
    
    # Ensure tensors are on the same device
    projection = sample.x_projection
    if mean.device != projection.device:
        mean = mean.to(projection.device)
    
    # Calculate the Euclidean distance
    distance_val = torch.sqrt(torch.sum((projection - mean) ** 2))
    return distance_val.item()


def factual_counterfactual_distance(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the Euclidean distance between the factual and counterfactual projected features.

    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance.

    Returns:
        The Euclidean distance.
    """
    # Check if both have projection attributes
    if (not hasattr(factual, 'x_projection') or not hasattr(counterfactual, 'x_projection') or
        factual.x_projection is None or counterfactual.x_projection is None):
        warnings.warn("One or both samples missing x_projection attribute")
        return 0.0
    
    # Ensure tensors are on the same device
    factual_proj = factual.x_projection
    counterfactual_proj = counterfactual.x_projection
    
    if factual_proj.device != counterfactual_proj.device:
        counterfactual_proj = counterfactual_proj.to(factual_proj.device)
    
    # Calculate the Euclidean distance
    distance_val = torch.sqrt(torch.sum((factual_proj - counterfactual_proj) ** 2))
    return distance_val.item()


# =============================================================================
# FIDELITY METRICS
# =============================================================================

def fidelity(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the fidelity of the counterfactual explanation.
    
    Fidelity measures how well the explanation changes the model's prediction
    from the factual to the desired counterfactual class.

    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance.

    Returns:
        The fidelity score (difference in prediction correctness).
    """
    try:
        # Node-level fidelity calculation
        if hasattr(factual, 'new_idx') and hasattr(counterfactual, 'sub_index'):
            factual_index = factual.new_idx
            counterfactual_index = counterfactual.sub_index
            
            phi_G = factual.y[factual_index]
            y = factual.y_ground[factual_index]
            phi_G_i = counterfactual.y[counterfactual_index]
            
        # Graph-level fidelity calculation
        else:
            phi_G = factual.y
            y = factual.y_ground if hasattr(factual, 'y_ground') else factual.y
            phi_G_i = counterfactual.y
        
        # Calculate fidelity components
        prediction_fidelity = 1 if phi_G == y else 0
        counterfactual_fidelity = 1 if phi_G_i == y else 0
        
        return prediction_fidelity - counterfactual_fidelity
        
    except Exception as e:
        warnings.warn(f"Error calculating fidelity: {e}")
        return 0.0


# =============================================================================
# SPARSITY METRICS
# =============================================================================

def edge_sparsity(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the edge sparsity between the factual and counterfactual graphs.
    
    Edge sparsity measures the fraction of edges that differ between the two graphs.

    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance.

    Returns:
        The edge sparsity (fraction of modified edges).
    """
    if (factual.edge_index is None or counterfactual.edge_index is None or
        factual.edge_index.numel() == 0):
        return 0.0
    
    try:
        # Get the edge indices as sets of tuples
        factual_edges = set(map(tuple, factual.edge_index.t().tolist()))
        counterfactual_edges = set(map(tuple, counterfactual.edge_index.t().tolist()))
        
        # Handle empty counterfactual edge set
        if not counterfactual_edges:
            return 1.0 if factual_edges else 0.0
        
        # Calculate the number of modified edges
        modified_edges = len(factual_edges.symmetric_difference(counterfactual_edges))
        
        # Calculate the total number of edges in the factual graph
        total_edges = len(factual_edges)
        
        # Calculate the edge sparsity
        if total_edges == 0:
            return 0.0
            
        sparsity = modified_edges / total_edges
        return sparsity
        
    except Exception as e:
        warnings.warn(f"Error calculating edge sparsity: {e}")
        return 0.0


def node_sparsity(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the node sparsity between the factual and counterfactual graphs.
    
    Node sparsity measures the fraction of node features that differ between the two graphs.

    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance.

    Returns:
        The node sparsity (fraction of modified node features).
    """
    if factual.x is None or counterfactual.x is None:
        return 0.0
    
    try:
        # Ensure tensors are on the same device
        factual_x = factual.x
        counterfactual_x = counterfactual.x
        
        if factual_x.device != counterfactual_x.device:
            counterfactual_x = counterfactual_x.to(factual_x.device)
        
        # Handle shape mismatches
        if factual_x.shape != counterfactual_x.shape:
            warnings.warn("Shape mismatch between factual and counterfactual node features")
            return 0.0
        
        # Calculate the number of modified node features
        modified_attributes = torch.sum(factual_x != counterfactual_x)
        
        # Calculate the node sparsity
        total_elements = factual_x.numel()
        if total_elements == 0:
            return 0.0
            
        sparsity = modified_attributes / total_elements
        return sparsity.item()
        
    except Exception as e:
        warnings.warn(f"Error calculating node sparsity: {e}")
        return 0.0


def edge_attr_sparsity(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the edge attribute sparsity between the factual and counterfactual graphs.
    
    Edge attribute sparsity measures the fraction of edge attributes that differ 
    between the two graphs.

    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance.

    Returns:
        The edge attribute sparsity (fraction of modified edge attributes).
    """
    if (not hasattr(factual, 'edge_attr') or not hasattr(counterfactual, 'edge_attr') or
        factual.edge_attr is None or counterfactual.edge_attr is None):
        return 0.0
    
    try:
        # Ensure tensors are on the same device
        factual_attr = factual.edge_attr
        counterfactual_attr = counterfactual.edge_attr
        
        if factual_attr.device != counterfactual_attr.device:
            counterfactual_attr = counterfactual_attr.to(factual_attr.device)
        
        # Handle shape mismatches
        if factual_attr.shape != counterfactual_attr.shape:
            warnings.warn("Shape mismatch between factual and counterfactual edge attributes")
            return 0.0
        
        # Calculate the number of modified edge attributes
        modified_attributes = torch.sum(factual_attr != counterfactual_attr)
        
        # Calculate the edge attribute sparsity
        total_elements = factual_attr.numel()
        if total_elements == 0:
            return 0.0
            
        sparsity = modified_attributes / total_elements
        return sparsity.item()
        
    except Exception as e:
        warnings.warn(f"Error calculating edge attribute sparsity: {e}")
        return 0.0


# =============================================================================
# GRAPH STRUCTURE METRICS
# =============================================================================

def graph_edit_distance(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the graph edit distance between the factual and counterfactual graphs.
    
    Graph edit distance counts the number of edge operations (additions/deletions)
    needed to transform one graph into another.

    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance.

    Returns:
        The graph edit distance (number of edge modifications).
    """
    if factual.edge_index is None:
        return 0.0
    
    try:
        # Get the edge indices as sets of tuples
        factual_edges = set(map(tuple, factual.edge_index.t().tolist()))
        
        # Handle case where counterfactual has no edges
        if (counterfactual.edge_index is None or 
            counterfactual.edge_index.numel() == 0):
            return float(len(factual_edges))
        
        counterfactual_edges = set(map(tuple, counterfactual.edge_index.t().tolist()))
        
        # Calculate the number of modified edges (symmetric difference)
        modified_edges = len(factual_edges.symmetric_difference(counterfactual_edges))
        
        return float(modified_edges)
        
    except Exception as e:
        warnings.warn(f"Error calculating graph edit distance: {e}")
        return 0.0


def perturbation_distance(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the perturbation distance between the factual and counterfactual node features.
    
    This metric computes the average Hamming distance per node for binary features.

    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance.

    Returns:
        The perturbation distance (average Hamming distance per node).
    """
    if factual.x is None or counterfactual.x is None:
        return 0.0
    
    try:
        # Ensure tensors are on the same device
        factual_x = factual.x
        counterfactual_x = counterfactual.x
        
        if factual_x.device != counterfactual_x.device:
            counterfactual_x = counterfactual_x.to(factual_x.device)
        
        # Handle shape mismatches
        if factual_x.shape != counterfactual_x.shape:
            warnings.warn("Shape mismatch between factual and counterfactual node features")
            return 0.0
        
        # Calculate the perturbation distance using XOR for binary features
        # Convert to long for XOR operation, then back to float for mean calculation
        xor_result = (factual_x.long() ^ counterfactual_x.long()).sum(dim=1).float()
        perturbation_dist = torch.mean(xor_result)
        
        return perturbation_dist.item()
        
    except Exception as e:
        warnings.warn(f"Error calculating perturbation distance: {e}")
        return 0.0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_all_metrics(
    factual: Data,
    counterfactual: Optional[Data] = None,
    mean_projection: Optional[torch.Tensor] = None,
    task_type: TaskType = TaskType.NODE
) -> Dict[str, float]:
    """
    Convenience function to compute all available metrics.
    
    Args:
        factual: The factual data instance.
        counterfactual: The counterfactual data instance (optional).
        mean_projection: Mean projection vector for distance calculations (optional).
        task_type: The type of explanation task.
        
    Returns:
        Dictionary containing all computed metrics.
    """
    calculator = MetricCalculator(task_type)
    return calculator.compute_all_metrics(factual, counterfactual, mean_projection)


# Backward compatibility - expose individual functions at module level
__all__ = [
    'MetricCalculator',
    'TaskType',
    'sample_distance_from_mean',
    'sample_distance_from_mean_projection', 
    'factual_counterfactual_distance',
    'fidelity',
    'edge_sparsity',
    'node_sparsity',
    'edge_attr_sparsity',
    'graph_edit_distance',
    'perturbation_distance',
    'compute_all_metrics'
]