"""
Graph Neural Network models for node, graph, and link prediction tasks.

This module contains various GNN architectures including GCN, ChebNet, GraphConv,
GAT, GINE, and SAGE models for node-level, graph-level, and link prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GraphConv, ChebConv, GATConv, GINEConv, SAGEConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from typing import Optional, Tuple, Union

from src.datasets.datainfo import DataInfo



class BaseGNN(nn.Module):
    """Base class for Graph Neural Networks with common functionality."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__()
        self.num_features = datainfo.num_features
        self.num_classes = datainfo.num_classes
        self.hidden_layers = cfg.model.hidden_layers
        self.dropout = cfg.model.dropout
        self.layers = nn.ModuleList()
        
    def _apply_layer_with_activation(self, x: torch.Tensor, layer: nn.Module, 
                                   edge_index: torch.Tensor, 
                                   edge_weights: Optional[torch.Tensor] = None,
                                   edge_attr: Optional[torch.Tensor] = None,
                                   activation: str = 'relu') -> torch.Tensor:
        """Apply a layer with optional edge weights/attributes and activation."""
        if hasattr(layer, 'edge_dim') and edge_attr is not None:
            # For layers that support edge attributes (like GINEConv)
            x = layer(x, edge_index, edge_attr)
        elif edge_weights is not None and hasattr(layer, '__call__'):
            # Check if layer accepts edge_weights parameter
            try:
                x = layer(x, edge_index, edge_weights=edge_weights)
            except TypeError:
                x = layer(x, edge_index)
        else:
            x = layer(x.float(), edge_index)
            
        # Apply activation function
        if activation == 'relu':
            x = F.relu(x)
        elif activation == 'elu':
            x = F.elu(x)
        elif activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif activation == 'tanh':
            x = torch.tanh(x)
        elif activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif activation == 'none':
            pass  # No activation
            
        return F.dropout(x, self.dropout, training=self.training)


class GCN(BaseGNN):
    """
    Graph Convolutional Network (GCN) for node classification.
    
    Implements a multi-layer GCN with skip connections through concatenation
    of all layer outputs.
    
    Args:
        datainfo (DataInfo): Dataset information containing features and classes.
        cfg: Configuration object with model parameters.
    """
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        
        # Build GCN layers
        self._build_layers()
        
        # Output layer with concatenated features from all layers
        self.output_layer = nn.Linear(sum(self.hidden_layers), self.num_classes)

    def _build_layers(self) -> None:
        """Build the GCN layers."""
        # Input layer
        self.layers.append(GCNConv(self.num_features, self.hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(GCNConv(self.hidden_layers[i-1], self.hidden_layers[i]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GCN.

        Args:
            x: Input feature matrix [num_nodes, num_features].
            edge_index: Edge indices [2, num_edges].
            edge_weights: Optional edge weights [num_edges].

        Returns:
            Log-softmax output [num_nodes, num_classes].
        """
        layer_outputs = []
        
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, edge_weights)
            layer_outputs.append(x)
        
        # Concatenate all layer outputs
        x = torch.cat(layer_outputs, dim=1)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding representation of the input.

        Args:
            x: Input feature matrix.
            edge_index: Edge indices.

        Returns:
            Final embedding representation.
        """
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index)
        return x


class ChebNet(BaseGNN):
    """
    Chebyshev Graph Convolutional Network with configurable polynomial order.
    
    Args:
        datainfo (DataInfo): Dataset information.
        cfg: Configuration object with model parameters including K (polynomial order).
    """
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self.k_order = getattr(cfg.model, 'K', 3)  # Default K=3
        self._build_layers()
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.num_classes)

    def _build_layers(self) -> None:
        """Build the Chebyshev convolution layers."""
        # Input layer
        self.layers.append(ChebConv(self.num_features, self.hidden_layers[0], K=self.k_order))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(ChebConv(self.hidden_layers[i-1], self.hidden_layers[i], K=self.k_order))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the ChebNet."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, edge_weights)
        
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get embedding representation."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index)
        return x


class GraphConvNet(BaseGNN):
    """
    Graph Convolutional Network using GraphConv layers.
    
    Args:
        datainfo (DataInfo): Dataset information.
        cfg: Configuration object with model parameters.
    """
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self._build_layers()
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.num_classes)

    def _build_layers(self) -> None:
        """Build the GraphConv layers."""
        # Input layer
        self.layers.append(GraphConv(self.num_features, self.hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(GraphConv(self.hidden_layers[i-1], self.hidden_layers[i]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the GraphConvNet."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, edge_weights)
        
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get embedding representation."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index)
        return x


class BaseGraphLevelGNN(BaseGNN):
    """Base class for graph-level prediction models."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        
    def _apply_global_pooling(self, x: torch.Tensor, batch: torch.Tensor, 
                            pooling_type: str = 'mean') -> torch.Tensor:
        """Apply global pooling for graph-level representation."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        if pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif pooling_type == 'add':
            return global_add_pool(x, batch)
        elif pooling_type == 'max':
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")


class GCN_G(BaseGraphLevelGNN):
    """GCN for graph classification with global pooling."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self._build_layers()
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.num_classes)

    def _build_layers(self) -> None:
        """Build the GCN layers."""
        # Input layer
        self.layers.append(GCNConv(self.num_features, self.hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(GCNConv(self.hidden_layers[i-1], self.hidden_layers[i]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None, 
                edge_weights: Optional[torch.Tensor] = None,
                edge_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with global pooling."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, edge_weights)
        
        # Global pooling
        x = self._apply_global_pooling(x, batch)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """Get graph-level embedding representation."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index)
        
        return self._apply_global_pooling(x, batch)


class ChebNet_G(BaseGraphLevelGNN):
    """ChebNet for graph classification."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self.k_order = getattr(cfg.model, 'K', 3)
        self._build_layers()
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.num_classes)

    def _build_layers(self) -> None:
        """Build the Chebyshev layers."""
        # Input layer
        self.layers.append(ChebConv(self.num_features, self.hidden_layers[0], K=self.k_order))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(ChebConv(self.hidden_layers[i-1], self.hidden_layers[i], K=self.k_order))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with global pooling."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, edge_weights)
        
        x = self._apply_global_pooling(x, batch)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """Get graph-level embedding representation."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index)
        
        return self._apply_global_pooling(x, batch)


class GraphConvNet_G(BaseGraphLevelGNN):
    """GraphConv for graph classification."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self._build_layers()
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.num_classes)

    def _build_layers(self) -> None:
        """Build the GraphConv layers."""
        # Input layer
        self.layers.append(GraphConv(self.num_features, self.hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(GraphConv(self.hidden_layers[i-1], self.hidden_layers[i]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None, 
                edge_weights: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with global pooling."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, edge_weights)
        
        x = self._apply_global_pooling(x, batch)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """Get graph-level embedding representation."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index)
        
        return self._apply_global_pooling(x, batch)


class GINENet_G(BaseGraphLevelGNN):
    """
    Graph Isomorphism Network with Edge features (GINE) for graph classification.
    
    Uses sum pooling which is typically preferred for GIN-based architectures.
    """
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self.edge_attr_dim = getattr(datainfo, 'edge_attr_dim', 0)
        self._build_layers()
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.num_classes)

    def _build_layers(self) -> None:
        """Build GINE layers with MLPs."""
        # Input layer with MLP
        input_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[0], self.hidden_layers[0])
        )
        self.layers.append(GINEConv(input_mlp, edge_dim=self.edge_attr_dim))

        # Hidden layers (using GCN for simplicity, but could be more GINE layers)
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(GCNConv(self.hidden_layers[i-1], self.hidden_layers[i]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using sum pooling for GINE."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GINEConv) and i == 0:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)
            
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Use sum pooling for GIN-based architectures
        x = self._apply_global_pooling(x, batch, pooling_type='add')
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          batch: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get graph-level embedding representation."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GINEConv) and i == 0:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)
            
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        return self._apply_global_pooling(x, batch, pooling_type='add')


class GAT_G(BaseGraphLevelGNN):
    """
    Graph Attention Network (GAT) for graph classification.
    
    Implements multi-head attention with configurable number of heads.
    """
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self.heads = getattr(cfg.model, 'heads', 8)
        self._build_layers()
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.num_classes)
        
    def _build_layers(self) -> None:
        """Build GAT layers with multi-head attention."""
        # Input layer with multi-head attention
        self.layers.append(GATConv(
            self.num_features, 
            self.hidden_layers[0] // self.heads, 
            heads=self.heads, 
            add_self_loops=True
        ))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(GATConv(
                self.hidden_layers[i-1], 
                self.hidden_layers[i] // self.heads, 
                heads=self.heads,
                add_self_loops=True
            ))
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using ELU activation for GAT."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, 
                                                edge_attr=edge_attr, activation='elu')
        
        x = self._apply_global_pooling(x, batch)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)
    
    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """Get graph-level embedding representation."""
        for layer in self.layers:
            x = self._apply_layer_with_activation(x, layer, edge_index, activation='elu')
        
        return self._apply_global_pooling(x, batch)


# =============================================================================
# LINK PREDICTION MODELS
# =============================================================================

class BaseLinkPredictor(nn.Module):
    """Base class for link prediction models."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__()
        self.num_features = datainfo.num_features
        self.hidden_layers = cfg.model.hidden_layers
        self.dropout = cfg.model.dropout
        self.prediction_method = getattr(cfg.model, 'prediction_method', 'dot_product')
        
    def _get_edge_embeddings(self, node_embeddings: torch.Tensor, 
                           edge_index: torch.Tensor, 
                           method: str = 'concat') -> torch.Tensor:
        """
        Get edge embeddings from node embeddings.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, embedding_dim].
            edge_index: Edge indices [2, num_edges].
            method: Method for combining node embeddings ('concat', 'hadamard', 'l1', 'l2').
            
        Returns:
            Edge embeddings [num_edges, edge_embedding_dim].
        """
        row, col = edge_index
        src_embeddings = node_embeddings[row]
        dst_embeddings = node_embeddings[col]
        
        if method == 'concat':
            return torch.cat([src_embeddings, dst_embeddings], dim=1)
        elif method == 'hadamard':
            return src_embeddings * dst_embeddings
        elif method == 'l1':
            return torch.abs(src_embeddings - dst_embeddings)
        elif method == 'l2':
            return (src_embeddings - dst_embeddings) ** 2
        elif method == 'sum':
            return src_embeddings + dst_embeddings
        elif method == 'mean':
            return (src_embeddings + dst_embeddings) / 2
        else:
            raise ValueError(f"Unknown edge embedding method: {method}")
    
    def _predict_links(self, node_embeddings: torch.Tensor, 
                      edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict link probabilities.
        
        Args:
            node_embeddings: Node embeddings.
            edge_index: Edge indices to predict.
            
        Returns:
            Link probabilities [num_edges].
        """
        row, col = edge_index
        src_embeddings = node_embeddings[row]
        dst_embeddings = node_embeddings[col]
        
        if self.prediction_method == 'dot_product':
            try:
                scores = (src_embeddings * dst_embeddings).sum(dim=1)
            except IndexError as e:
                scores = (src_embeddings * dst_embeddings).unsqueeze(0).sum(dim=1)
        elif self.prediction_method == 'cosine':
            scores = F.cosine_similarity(src_embeddings, dst_embeddings)
        elif self.prediction_method == 'mlp':
            edge_embeddings = self._get_edge_embeddings(
                node_embeddings, edge_index, method='concat'
            )
            scores = self.link_predictor(edge_embeddings).squeeze()
        else:
            raise ValueError(f"Unknown prediction method: {self.prediction_method}")
        
        return torch.sigmoid(scores)


class GCNLinkPredictor(BaseLinkPredictor):
    """GCN-based link prediction model."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self._build_encoder()
        
        if self.prediction_method == 'mlp':
            self._build_link_predictor()
    
    def _build_encoder(self) -> None:
        """Build the GCN encoder."""
        self.encoder_layers = nn.ModuleList()
        
        # Input layer
        self.encoder_layers.append(GCNConv(self.num_features, self.hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.encoder_layers.append(GCNConv(self.hidden_layers[i-1], self.hidden_layers[i]))
    
    def _build_link_predictor(self) -> None:
        """Build MLP for link prediction."""
        input_dim = self.hidden_layers[-1] * 2  # Concatenated embeddings
        self.link_predictor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_layers[-1]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layers[-1], self.hidden_layers[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layers[-1] // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                predict_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for link prediction.
        
        Args:
            x: Node features [num_nodes, num_features].
            edge_index: Training edge indices [2, num_train_edges].
            predict_edge_index: Edge indices to predict [2, num_predict_edges].
            
        Returns:
            Link probabilities [num_predict_edges].
        """
        # Encode nodes using GCN
        node_embeddings = self.encode(x, edge_index)
        
        # Predict links
        return self._predict_links(node_embeddings, predict_edge_index)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes into embeddings."""
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:  # No activation on last layer
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        return self.encode(x, edge_index)


class SAGELinkPredictor(BaseLinkPredictor):
    """GraphSAGE-based link prediction model."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self._build_encoder()
        
        if self.prediction_method == 'mlp':
            self._build_link_predictor()
    
    def _build_encoder(self) -> None:
        """Build the SAGE encoder."""
        self.encoder_layers = nn.ModuleList()
        
        # Input layer
        self.encoder_layers.append(SAGEConv(self.num_features, self.hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.encoder_layers.append(SAGEConv(self.hidden_layers[i-1], self.hidden_layers[i]))
    
    def _build_link_predictor(self) -> None:
        """Build MLP for link prediction."""
        input_dim = self.hidden_layers[-1] * 2  # Concatenated embeddings
        self.link_predictor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_layers[-1]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layers[-1], self.hidden_layers[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layers[-1] // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                predict_edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for link prediction."""
        # Encode nodes using SAGE
        node_embeddings = self.encode(x, edge_index)
        
        # Predict links
        return self._predict_links(node_embeddings, predict_edge_index)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes into embeddings."""
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:  # No activation on last layer
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        return self.encode(x, edge_index)


class GATLinkPredictor(BaseLinkPredictor):
    """GAT-based link prediction model."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self.heads = getattr(cfg.model, 'heads', 8)
        self._build_encoder()
        
        if self.prediction_method == 'mlp':
            self._build_link_predictor()
    
    def _build_encoder(self) -> None:
        """Build the GAT encoder."""
        self.encoder_layers = nn.ModuleList()
        
        # Input layer with multi-head attention
        self.encoder_layers.append(GATConv(
            self.num_features, 
            self.hidden_layers[0] // self.heads, 
            heads=self.heads,
            add_self_loops=True
        ))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            # Last layer uses single head
            heads = 1 if i == len(self.hidden_layers) - 1 else self.heads
            out_channels = self.hidden_layers[i] if i == len(self.hidden_layers) - 1 else self.hidden_layers[i] // heads
            
            self.encoder_layers.append(GATConv(
                self.hidden_layers[i-1], 
                out_channels,
                heads=heads,
                add_self_loops=True
            ))
    
    def _build_link_predictor(self) -> None:
        """Build MLP for link prediction."""
        input_dim = self.hidden_layers[-1] * 2  # Concatenated embeddings
        self.link_predictor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_layers[-1]),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layers[-1], self.hidden_layers[-1] // 2),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layers[-1] // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                predict_edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for link prediction."""
        # Encode nodes using GAT
        node_embeddings = self.encode(x, edge_index)
        
        # Predict links
        return self._predict_links(node_embeddings, predict_edge_index)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes into embeddings."""
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:  # No activation on last layer
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        return self.encode(x, edge_index)


class VariationalAutoEncoderLinkPredictor(BaseLinkPredictor):
    """Variational Graph Auto-Encoder for link prediction."""
    
    def __init__(self, datainfo: DataInfo, cfg):
        super().__init__(datainfo, cfg)
        self.latent_dim = getattr(cfg.model, 'latent_dim', self.hidden_layers[-1])
        self._build_encoder()
    
    def _build_encoder(self) -> None:
        """Build the encoder with mean and log variance outputs."""
        self.encoder_layers = nn.ModuleList()
        
        # Shared encoder layers
        self.encoder_layers.append(GCNConv(self.num_features, self.hidden_layers[0]))
        for i in range(1, len(self.hidden_layers) - 1):
            self.encoder_layers.append(GCNConv(self.hidden_layers[i-1], self.hidden_layers[i]))
        
        # Mean and log variance layers
        self.mean_layer = GCNConv(self.hidden_layers[-2], self.latent_dim)
        self.logvar_layer = GCNConv(self.hidden_layers[-2], self.latent_dim)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to mean and log variance."""
        # Shared encoding
        for layer in self.encoder_layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Get mean and log variance
        mean = self.mean_layer(x, edge_index)
        logvar = self.logvar_layer(x, edge_index)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                predict_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predictions, mean, and log variance.
        
        Returns:
            Tuple of (link_probabilities, mean, logvar) for loss computation.
        """
        mean, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mean, logvar)
        
        # Predict links
        link_probs = self._predict_links(z, predict_edge_index)
        
        return link_probs, mean, logvar
    
    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings (mean of the latent distribution)."""
        mean, _ = self.encode(x, edge_index)
        return mean


# Model registry for easy instantiation
MODEL_REGISTRY = {
    # Node-level models
    'GCN': GCN,
    'ChebNet': ChebNet, 
    'GraphConvNet': GraphConvNet,
    
    # Graph-level models
    'GCN_G': GCN_G,
    'ChebNet_G': ChebNet_G,
    'GraphConvNet_G': GraphConvNet_G,
    'GINENet_G': GINENet_G,
    'GAT_G': GAT_G,
    
    # Link prediction models
    'GCNLinkPredictor': GCNLinkPredictor,
    'SAGELinkPredictor': SAGELinkPredictor,
    'GATLinkPredictor': GATLinkPredictor,
    'VAELinkPredictor': VariationalAutoEncoderLinkPredictor,
}


def create_model(model_name: str, datainfo: DataInfo, cfg) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model to create.
        datainfo: Dataset information.
        cfg: Configuration object.
        
    Returns:
        Instantiated model.
        
    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](datainfo, cfg)


def get_link_prediction_models() -> list:
    """Get list of available link prediction models."""
    return [name for name in MODEL_REGISTRY.keys() if 'LinkPredictor' in name or 'VAE' in name]


def get_node_level_models() -> list:
    """Get list of available node-level models."""
    return [name for name in MODEL_REGISTRY.keys() if not name.endswith('_G') and 'LinkPredictor' not in name and 'VAE' not in name]


def get_graph_level_models() -> list:
    """Get list of available graph-level models."""
    return [name for name in MODEL_REGISTRY.keys() if name.endswith('_G')]