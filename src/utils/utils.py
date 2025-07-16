from __future__ import annotations

import logging
import sys
import time
import warnings
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from texttable import Texttable
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph, to_dense_adj
from typing import Any, Dict
from omegaconf import DictConfig, DictKeyType
import yaml

def visualize_graph(data, ax, title):
    """Visualize a single graph from the dataset using node features and edge attributes"""
    # Convert to NetworkX graph
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    
    # Add nodes with features
    for i in range(data.num_nodes):
        G.add_node(i)
    
    # Add edges with attributes if available
    edges = [(edge_index[0][i], edge_index[1][i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Handle node features for coloring
    if data.x is not None and data.x.size(1) > 0:
        # Use first feature dimension for node coloring
        node_features = data.x[:, 0].numpy()
        node_colors = node_features
        cmap = cm.viridis
        norm = Normalize(vmin=node_features.min(), vmax=node_features.max())
        
        # Calculate node sizes based on feature magnitude if multiple features
        if data.x.size(1) > 1:
            node_sizes = np.abs(data.x.sum(dim=1).numpy()) * 20 + 30
        else:
            node_sizes = 50
    else:
        node_colors = 'lightblue'
        node_sizes = 50
        cmap = None
        norm = None
    
    # Handle edge attributes for edge widths/colors
    edge_widths = 1.5
    edge_colors = 'black'
    edge_collection = None
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        if data.edge_attr.size(1) > 0:
            # Use first edge attribute for edge width
            edge_attrs = data.edge_attr[:, 0].numpy()
            edge_widths = np.abs(edge_attrs) * 2 + 0.5
            
            # Use edge attributes for coloring if more than one dimension
            if data.edge_attr.size(1) > 1:
                edge_colors = data.edge_attr[:, 1].numpy()
                # Normalize edge colors to [0, 1] range for proper color mapping
                if edge_colors.max() != edge_colors.min():
                    edge_colors = (edge_colors - edge_colors.min()) / (edge_colors.max() - edge_colors.min())
                else:
                    edge_colors = np.ones_like(edge_colors) * 0.5
    
    # Draw the graph
    if isinstance(edge_colors, np.ndarray):
        # Use matplotlib's LineCollection for proper color mapping with edge attributes
        from matplotlib.collections import LineCollection
        
        # Create edge segments for LineCollection
        edge_segments = []
        for edge in G.edges():
            edge_segments.append([pos[edge[0]], pos[edge[1]]])
        
        # Create LineCollection with proper color mapping
        line_collection = LineCollection(edge_segments, 
                                       linewidths=edge_widths,
                                       colors=cm.plasma(edge_colors),
                                       alpha=0.7)
        ax.add_collection(line_collection)
        edge_collection = line_collection
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                             edge_color=edge_colors, alpha=0.7)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                                  node_size=node_sizes, cmap=cmap, alpha=0.8)
    
    # Add colorbar for node features if applicable
    if cmap is not None and norm is not None:
        plt.colorbar(nodes, ax=ax, shrink=0.8, label='Node Feature Value')
    
    # Add colorbar for edge features if applicable
    if isinstance(edge_colors, np.ndarray) and edge_collection is not None:
        # Create a ScalarMappable for the edge colorbar
        from matplotlib.cm import ScalarMappable
        # Don't import Normalize here since it's already imported at the top
        
        edge_norm = Normalize(vmin=edge_colors.min(), vmax=edge_colors.max())
        edge_sm = ScalarMappable(norm=edge_norm, cmap=cm.plasma)
        plt.colorbar(edge_sm, ax=ax, shrink=0.6, label='Edge Feature Value', 
                    orientation='horizontal', pad=0.1)
    
    # Create title with feature information
    feature_info = ""
    if data.x is not None:
        feature_info += f"Node features: {data.x.size(1)}"
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        feature_info += f", Edge features: {data.edge_attr.size(1)}"
    
    ax.set_title(f'{title}\nNodes: {data.num_nodes}, Edges: {data.num_edges}\n{feature_info}', 
                fontsize=9)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("Prova.jpeg")


def print_info(string: str) -> None:
    from datetime import datetime
    
    """
    Print a formatted string with additional arguments.
    
    Parameters:
    - string (str): The format string.
    - *args (Any): Additional arguments to format the string.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {string}")


def merge_dict(dict_1: dict | DictConfig, dict_2: dict):

    dict_1.update({key: dict_2[key] for key in dict_1 if key in dict_2})
    

def merge_hydra_wandb(cfg, wandb):
    
    for k, v in cfg.items():
        if type(v) == DictConfig:
              
            merge_dict(v, wandb)
    

def read_yaml(filename):
    
    with open(filename, 'r') as file:
        return yaml.safe_load(file)
    
    
    
def flatten_dict(d: Dict[Any, Any] | DictConfig, parent_key: str | DictKeyType = '', sep: str = '_') -> dict:
    
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


"""
Unified Utilities for COMBINEX Explainer.

This module provides utility functions for both node-level and graph-level
explanation tasks, including graph construction, visualization, optimization,
and tensor operations.
"""




# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class TimeOutException(Exception):
    """Custom exception for timeout scenarios."""
    
    def __init__(self, message: str = "Operation timed out", *args) -> None:
        super().__init__(message, *args)
        self.message = message


# =============================================================================
# DECORATORS
# =============================================================================

def timeit(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to be timed.
        
    Returns:
        Wrapped function with timing capability.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def safe_execution(default_return=None, log_errors=True):
    """
    Decorator for safe function execution with error handling.
    
    Args:
        default_return: Default value to return on error.
        log_errors: Whether to log errors.
        
    Returns:
        Wrapped function with error handling.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.getLogger(func.__module__).error(
                        f"Error in {func.__name__}: {e}"
                    )
                return default_return
        return wrapper
    return decorator


# =============================================================================
# OPTIMIZATION UTILITIES
# =============================================================================

class OptimizerFactory:
    """Factory class for creating optimizers."""
    
    @staticmethod
    def create_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            cfg: Configuration object.
            model: Model to optimize.
            
        Returns:
            Configured optimizer.
            
        Raises:
            ValueError: If optimizer name is not supported.
        """
        optimizer_name = cfg.optimizer.name.lower()
        lr = cfg.optimizer.lr
        
        optimizer_map = {
            "sgd": lambda: OptimizerFactory._create_sgd(cfg, model),
            "adadelta": lambda: optim.Adadelta(model.parameters(), lr=lr),
            "adam": lambda: optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.0)),
            "adamw": lambda: optim.AdamW(model.parameters(), lr=lr),
            "rmsprop": lambda: optim.RMSprop(model.parameters(), lr=lr),
        }
        
        if optimizer_name not in optimizer_map:
            available = list(optimizer_map.keys())
            raise ValueError(
                f"Optimizer '{optimizer_name}' not supported. "
                f"Available optimizers: {available}"
            )
        
        return optimizer_map[optimizer_name]()
    
    @staticmethod
    def _create_sgd(cfg, model: nn.Module) -> optim.SGD:
        """Create SGD optimizer with optional momentum."""
        lr = cfg.optimizer.lr
        momentum = getattr(cfg.optimizer, 'n_momentum', 0.0)
        
        if momentum == 0.0:
            return optim.SGD(model.parameters(), lr=lr)
        else:
            return optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=momentum, 
                nesterov=True
            )


# Backward compatibility
def get_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    """Legacy function for backward compatibility."""
    return OptimizerFactory.create_optimizer(cfg, model)


# =============================================================================
# TENSOR OPERATIONS
# =============================================================================

class TensorUtils:
    """Utility class for tensor operations."""
    
    @staticmethod
    def get_degree_matrix(adj: Tensor) -> Tensor:
        """
        Compute degree matrix from adjacency matrix.
        
        Args:
            adj: Adjacency matrix.
            
        Returns:
            Degree matrix.
        """
        return torch.diag(adj.sum(dim=1))
    
    @staticmethod
    def normalize_adjacency(adj: Tensor) -> Tensor:
        """
        Normalize adjacency matrix using the GCN normalization trick.
        
        Args:
            adj: Adjacency matrix.
            
        Returns:
            Normalized adjacency matrix.
        """
        # Add self-loops
        A_tilde = adj + torch.eye(
            adj.size(0), 
            device=adj.device, 
            dtype=adj.dtype
        )
        
        # Compute degree matrix and its inverse square root
        D_tilde = torch.pow(TensorUtils.get_degree_matrix(A_tilde), -0.5)
        D_tilde[torch.isinf(D_tilde)] = 0  # Handle inf values
        
        # Normalized adjacency matrix
        return D_tilde @ A_tilde @ D_tilde
    
    @staticmethod
    def discretize_tensor(tensor: Tensor, mode: str = "threshold") -> Tensor:
        """
        Discretize tensor values.
        
        Args:
            tensor: Input tensor.
            mode: Discretization mode ('threshold' or 'round').
            
        Returns:
            Discretized tensor.
        """
        if mode == "threshold":
            return torch.where(
                tensor <= -0.5, -1, 
                torch.where(tensor > 0.5, 1, 0)
            )
        elif mode == "round":
            return torch.round(tensor)
        else:
            raise ValueError(f"Unknown discretization mode: {mode}")
    
    @staticmethod
    def discretize_to_nearest_integer(tensor: Tensor) -> Tensor:
        """
        Discretize tensor to nearest integer values.
        
        Args:
            tensor: Input tensor.
            
        Returns:
            Discretized tensor.
        """
        return TensorUtils.discretize_tensor(tensor, mode="round")
    
    @staticmethod
    def create_symmetric_matrix_from_vector(
        vector: Tensor, 
        n_rows: int, 
        device: str = "cpu"
    ) -> Tensor:
        """
        Create symmetric matrix from vector using lower triangular indices.
        
        Args:
            vector: Input vector.
            n_rows: Number of rows/columns.
            device: Device for computation.
            
        Returns:
            Symmetric matrix.
        """
        matrix = torch.zeros(n_rows, n_rows, device=device)
        idx = torch.tril_indices(n_rows, n_rows, device=device)
        matrix[idx[0], idx[1]] = vector
        return torch.tril(matrix) + torch.tril(matrix, -1).t()
    
    @staticmethod
    def create_vector_from_symmetric_matrix(matrix: Tensor) -> Tensor:
        """
        Extract vector from symmetric matrix using lower triangular indices.
        
        Args:
            matrix: Input symmetric matrix.
            
        Returns:
            Vector from lower triangle.
        """
        idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
        return matrix[idx[0], idx[1]]
    
    @staticmethod
    def index_to_mask(index: Tensor, size: int) -> Tensor:
        """
        Convert indices to boolean mask.
        
        Args:
            index: Tensor of indices.
            size: Size of the mask.
            
        Returns:
            Boolean mask tensor.
        """
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = True
        return mask


# Backward compatibility functions
get_degree_matrix = TensorUtils.get_degree_matrix
normalize_adj = TensorUtils.normalize_adjacency
discretize_tensor = TensorUtils.discretize_tensor
discretize_to_nearest_integer = TensorUtils.discretize_to_nearest_integer
create_symm_matrix_from_vec = TensorUtils.create_symmetric_matrix_from_vector
create_vec_from_symm_matrix = TensorUtils.create_vector_from_symmetric_matrix
index_to_mask = TensorUtils.index_to_mask


# =============================================================================
# GRAPH OPERATIONS
# =============================================================================

class GraphUtils:
    """Utility class for graph operations."""
    
    @staticmethod
    def get_neighbourhood(
        node_idx: int, 
        edge_index: Tensor, 
        n_hops: int, 
        features: Tensor, 
        labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[int, int]]:
        """
        Extract k-hop neighborhood of a node.
        
        Args:
            node_idx: Index of the central node.
            edge_index: Edge indices of the graph.
            n_hops: Number of hops.
            features: Node features.
            labels: Node labels.
            
        Returns:
            Tuple of (edge_index, features, labels, node_mapping).
        """
        # Get k-hop subgraph
        subset_nodes, subset_edge_index, _, _ = k_hop_subgraph(
            node_idx, n_hops, edge_index
        )
        
        # Relabel nodes
        edge_subset_relabel, _ = subgraph(
            subset_nodes, edge_index, relabel_nodes=True
        )
        
        # Extract features and labels
        sub_features = features[subset_nodes]
        sub_labels = labels[subset_nodes]
        
        # Create node mapping
        node_mapping = {
            int(original_idx): new_idx 
            for new_idx, original_idx in enumerate(subset_nodes.numpy())
        }
        
        return edge_subset_relabel, sub_features, sub_labels, node_mapping
    
    @staticmethod
    def check_graph_validity(edge_index: Tensor) -> bool:
        """
        Check if graph is empty or trivial.
        
        Args:
            edge_index: Edge index tensor.
            
        Returns:
            True if graph is empty/trivial, False otherwise.
        """
        return edge_index.size(1) <= 1
    
    @staticmethod
    def build_factual_graph(
        mask_index: int, 
        data: Data, 
        n_hops: int, 
        oracle: nn.Module, 
        predicted_labels: Tensor, 
        target_labels: Tensor, 
        device: str = "cuda"
    ) -> Data:
        """
        Build factual graph for node-level explanation.
        
        Args:
            mask_index: Index of the target node.
            data: Original graph data.
            n_hops: Number of hops for neighborhood.
            oracle: Oracle model.
            predicted_labels: Predicted labels.
            target_labels: Target labels.
            device: Computation device.
            
        Returns:
            Factual graph data.
        """
        # Get neighborhood
        sub_edge_index, sub_x, sub_labels, node_dict = GraphUtils.get_neighbourhood(
            node_idx=int(mask_index),
            edge_index=data.edge_index.cpu(),
            n_hops=n_hops,
            features=data.x.cpu(),
            labels=data.y.cpu()
        )
        
        # Get new index and subset
        new_idx = node_dict[int(mask_index)]
        sub_index = list(node_dict.keys())
        
        # Get labels for subset
        sub_y = predicted_labels[sub_index]
        sub_targets = target_labels[sub_index]
        
        # Get embedding representation
        oracle = oracle.to(device)
        with torch.no_grad():
            repr_tensor = oracle.get_embedding_repr(
                sub_x.to(device), sub_edge_index.to(device)
            ).detach()
            embedding_repr = torch.mean(repr_tensor, dim=0)
        
        # Create factual graph
        return Data(
            x=sub_x.cpu(),
            edge_index=sub_edge_index.cpu(),
            y=sub_y.cpu(),
            y_ground=sub_labels.cpu(),
            new_idx=new_idx,
            targets=sub_targets.cpu(),
            node_dict=node_dict,
            x_projection=embedding_repr.cpu()
        )
    
    @staticmethod
    def build_counterfactual_graph(
        x: Tensor, 
        edge_index: Tensor, 
        graph: Data, 
        oracle: nn.Module, 
        output_actual: Tensor, 
        device: str = "cuda"
    ) -> Data:
        """
        Build counterfactual graph for node-level explanation.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            graph: Original graph.
            oracle: Oracle model.
            output_actual: Model output.
            device: Computation device.
            
        Returns:
            Counterfactual graph data.
        """
        with torch.no_grad():
            embedding_repr = torch.mean(
                oracle.get_embedding_repr(x, edge_index), dim=0
            )
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.argmax(output_actual, dim=1),
            sub_index=graph.new_idx,
            x_projection=embedding_repr
        )
    
    @staticmethod
    def build_counterfactual_graph_gc(
        x: Tensor, 
        edge_index: Tensor, 
        graph: Data, 
        oracle: nn.Module, 
        output_actual: Tensor, 
        edge_attr: Optional[Tensor] = None,
        device: str = "cuda"
    ) -> Data:
        """
        Build counterfactual graph for graph-level explanation.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            graph: Original graph.
            oracle: Oracle model.
            output_actual: Model output.
            edge_attr: Optional edge attributes.
            device: Computation device.
            
        Returns:
            Counterfactual graph data.
        """
        # Prepare input dictionary
        input_dict = {
            "x": x,
            "edge_index": edge_index,
            "batch": graph.batch
        }
        
        if edge_attr is not None:
            input_dict["edge_attr"] = edge_attr
        
        # Get embedding representation
        with torch.no_grad():
            embedding_repr = torch.mean(
                oracle.get_embedding_repr(**input_dict), dim=0
            )
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.argmax(output_actual, dim=1),
            x_projection=embedding_repr
        )


# Backward compatibility functions
get_neighbourhood = GraphUtils.get_neighbourhood
check_graphs = GraphUtils.check_graph_validity
build_factual_graph = GraphUtils.build_factual_graph
build_counterfactual_graph = GraphUtils.build_counterfactual_graph
build_counterfactual_graph_gc = GraphUtils.build_counterfactual_graph_gc


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

class VisualizationUtils:
    """Utility class for graph visualization."""
    
    @staticmethod
    def plot_factual_and_counterfactual_graphs(
        factual_graph: Data, 
        counterfactual_graph: Data, 
        folder: str, 
        pid: int,
        save_dir: str = "data/artifacts"
    ) -> None:
        """
        Plot factual and counterfactual graphs side by side.
        
        Args:
            factual_graph: Factual graph data.
            counterfactual_graph: Counterfactual graph data.
            folder: Folder name for saving.
            pid: Process ID for unique naming.
            save_dir: Base directory for saving plots.
        """
        def plot_single_graph(graph: Data, title: str, ax) -> Dict[int, Tuple[float, float]]:
            """Plot a single graph and return node positions."""
            if graph.edge_index.size(1) == 0:
                ax.text(0.5, 0.5, "Empty Graph", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return {}
            
            # Convert to NetworkX graph
            edge_list = graph.edge_index.t().tolist()
            G = nx.Graph()
            G.add_edges_from(edge_list)
            
            # Create node positions
            if graph.x.shape[1] >= 2:
                # Use last two features as coordinates
                pos = {
                    i: (graph.x[i, -2].item(), graph.x[i, -1].item()) 
                    for i in range(graph.num_nodes)
                }
            else:
                # Use spring layout if no coordinate features
                pos = nx.spring_layout(G)
            
            # Create node labels
            if graph.x.shape[1] >= 4:
                node_labels = {
                    i: f"{graph.x[i, 0]:.0f}, {graph.x[i, 1]:.0f}\n"
                       f"{graph.x[i, 2]:.2f}, {graph.x[i, 3]:.2f}"
                    for i in range(graph.num_nodes)
                }
            else:
                node_labels = {i: str(i) for i in range(graph.num_nodes)}
            
            # Draw graph
            nx.draw(
                G, pos, labels=node_labels, with_labels=True,
                node_color='skyblue', edge_color='gray', node_size=500,
                font_size=8, font_weight='bold', ax=ax
            )
            
            ax.set_title(title)
            return pos
        
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot both graphs
        pos_factual = plot_single_graph(factual_graph, "Factual", axs[0])
        pos_counterfactual = plot_single_graph(counterfactual_graph, "Counterfactual", axs[1])
        
        # Set common axis limits
        if pos_factual and pos_counterfactual:
            all_positions = list(pos_factual.values()) + list(pos_counterfactual.values())
            if all_positions:
                all_x = [pos[0] for pos in all_positions]
                all_y = [pos[1] for pos in all_positions]
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                padding = 2
                
                for ax in axs:
                    ax.set_xlim(x_min - padding, x_max + padding)
                    ax.set_ylim(y_min - padding, y_max + padding)
        
        # Adjust layout and save
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
        plt.tight_layout()
        
        # Create save directory
        save_path = Path(save_dir) / folder
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        plt.savefig(save_path / f'factual_and_counterfactual_graphs_{pid}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


# Backward compatibility
plot_factual_and_counterfactual_graphs = (
    VisualizationUtils.plot_factual_and_counterfactual_graphs
)


@dataclass
class ExplanationContext:
    """Context object containing explanation results and metadata."""
    y_pred_new_actual: torch.Tensor
    target: torch.Tensor
    loss: float
    best_loss: float
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor]
    graph: Data
    oracle: Any
    output_actual: torch.Tensor
    device: str    

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

class DataUtils:
    """Utility class for data processing operations."""
    
    @staticmethod
    def get_s_values(pickled_results, header) -> pd.DataFrame:
        """
        Process pickled results into DataFrame.
        
        Args:
            pickled_results: List of pickled results.
            header: Column headers.
            
        Returns:
            Processed DataFrame.
        """
        df_prep = [
            example[0] for example in pickled_results 
            if example and len(example) > 0
        ]
        return pd.DataFrame(df_prep, columns=header)
    
    @staticmethod
    def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx) -> None:
        """
        Reformat dataset for PGExplainer compatibility.
        
        Args:
            dataset: Dataset to reformat.
            train_idx: Training indices.
            test_idx: Test indices.
        """
        dataset.data.train_mask = TensorUtils.index_to_mask(
            train_idx, size=dataset.data.num_nodes
        )
        dataset.data.test_mask = TensorUtils.index_to_mask(
            test_idx, size=dataset.data.num_nodes
        )
    
    @staticmethod
    def print_info(dictionary: Dict[str, Any]) -> None:
        """
        Print information in a formatted table.
        
        Args:
            dictionary: Dictionary of information to display.
        """
        if not dictionary:
            return
        
        # Initialize table
        table = Texttable(max_width=0)
        num_cols = len(dictionary)
        
        # Set table properties
        table.set_cols_align(["c"] * num_cols)
        table.set_cols_dtype(["t"] * num_cols)
        table.set_cols_valign(["m"] * num_cols)
        
        # Add data
        table.add_rows([
            list(dictionary.keys()),
            list(dictionary.values())
        ])
        
        # Clear screen and print
        sys.stdout.write("\033[H\033[J")  # Move cursor to top and clear screen
        sys.stdout.write(table.draw() + "\n")
        sys.stdout.flush()


# Backward compatibility functions
get_S_values = DataUtils.get_s_values
redo_dataset_pgexplainer_format = DataUtils.redo_dataset_pgexplainer_format
print_info = DataUtils.print_info

def get_trainer(task: str) -> Any:
    
    if task.lower() == "node":
        from src.oracles.train.train import Trainer
        return Trainer
    elif task.lower() == "graph":
        from src.oracles.train.train import GraphTrainer
        return GraphTrainer
    elif task.lower() == "link":
        from src.oracles.train.train import LinkPredictionTrainer
        return LinkPredictionTrainer


def get_graph_builder(task: str) -> Any:
    """
    Get the graph builder function.
    
    Returns:
        Function for building graphs.
    """
    
    task = task.lower()
    
    if task == "node":
        
        return GraphUtils.build_counterfactual_graph
    
    elif task == "graph":
        return GraphUtils.build_counterfactual_graph_gc


def check_function_factory(task: str, mode: str = None) -> Any:
    """
    Get the counterfactual condition function.
    
    Args:
        task: Task type ('node' or 'graph').
        mode: Mode for counterfactual condition.
        
    Returns:
        Function for checking counterfactual conditions.
    """
    task = task.lower()
    if task == "node" or task == "graph" or task == "link":
        
        def check_counterfactual_condition(
            loss: Tensor,
            best_loss: float,
            target: Data, 
            y_pred_new_actual: nn.Module
        ) -> bool:

            return y_pred_new_actual == target and loss < best_loss
        
        
        return check_counterfactual_condition
                
    else:
        raise ValueError(f"Unknown task type: {task}")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    'TimeOutException',
    
    # Decorators
    'timeit',
    'safe_execution',
    
    # Classes
    'OptimizerFactory',
    'TensorUtils',
    'GraphUtils',
    'VisualizationUtils',
    'DataUtils',
    
    # Backward compatibility functions
    'get_optimizer',
    'get_degree_matrix',
    'normalize_adj',
    'discretize_tensor',
    'discretize_to_nearest_integer',
    'create_symm_matrix_from_vec',
    'create_vec_from_symm_matrix',
    'index_to_mask',
    'get_neighbourhood',
    'check_graphs',
    'build_factual_graph',
    'build_counterfactual_graph',
    'build_counterfactual_graph_gc',
    'plot_factual_and_counterfactual_graphs',
    'get_S_values',
    'redo_dataset_pgexplainer_format',
    'print_info',
]