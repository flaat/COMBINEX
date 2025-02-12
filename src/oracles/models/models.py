import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, ChebConv, global_mean_pool


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) with multiple hidden layers.

    Args:
        num_features (int): Number of input features.
        hidden_layers (List[int]): List of hidden layer sizes.
        num_classes (int): Number of output classes.
        dropout (float, optional): Dropout rate. Default is 0.5.
    """
    def __init__(self, num_features: int, num_classes: int, cfg):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = cfg.model.dropout

        # Input layer
        self.layers.append(GCNConv(num_features, cfg.model.hidden_layers[0]))

        # Hidden layers
        for i in range(1, len(cfg.model.hidden_layers)):
            self.layers.append(GCNConv(cfg.model.hidden_layers[i-1], cfg.model.hidden_layers[i]))

        # Output layer
        self.output_layer = nn.Linear(sum(cfg.model.hidden_layers), num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the GCN.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.
            edge_weight (torch.Tensor, optional): Edge weights. Default is None.

        Returns:
            torch.Tensor: Log-softmax output.
        """
        layer_outputs = []
        for layer in self.layers:
            if edge_weight is not None:
                x = F.relu(layer(x, edge_index, edge_weight=edge_weight))
            else:
                x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
            layer_outputs.append(x)
        
        x = torch.cat(layer_outputs, dim=1)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding representation of the input.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Embedding representation.
        """
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        return x
   
    
class ChebNet(nn.Module):
    """
    Chebyshev Graph Convolutional Network (ChebNet) with multiple hidden layers.

    Args:
        num_features (int): Number of input features.
        hidden_layers (List[int]): List of hidden layer sizes.
        num_classes (int): Number of output classes.
        K (int, optional): Chebyshev polynomial order. Default is 3.
        dropout (float, optional): Dropout rate. Default is 0.5.
    """
    def __init__(self, num_features: int, num_classes: int, cfg):
        super(ChebNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = cfg.model.dropout

        # Input layer
        self.layers.append(ChebConv(num_features, cfg.model.hidden_layers[0], K=cfg.model.K))

        # Hidden layers
        for i in range(1, len(cfg.model.hidden_layers)):
            self.layers.append(ChebConv(cfg.model.hidden_layers[i-1], cfg.model.hidden_layers[i], K=cfg.model.K))

        # Output layer
        self.output_layer = nn.Linear(cfg.model.hidden_layers[-1], num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the ChebNet.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.
            edge_weights (torch.Tensor, optional): Edge weights. Default is None.

        Returns:
            torch.Tensor: Log-softmax output.
        """
        for layer in self.layers:
            if edge_weights is not None:
                x = F.relu(layer(x, edge_index, edge_weights))
            else:
                x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding representation of the input.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Embedding representation.
        """
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    

class GraphConvNet(nn.Module):
    """
    Graph Convolutional Network (GraphConvNet) with multiple hidden layers using GraphConv.

    Args:
        num_features (int): Number of input features.
        hidden_layers (List[int]): List of hidden layer sizes.
        num_classes (int): Number of output classes.
        dropout (float, optional): Dropout rate. Default is 0.5.
    """
    def __init__(self, num_features: int, num_classes: int, cfg):
        super(GraphConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = cfg.model.dropout

        # Input layer
        self.layers.append(GraphConv(num_features, cfg.model.hidden_layers[0]))

        # Hidden layers
        for i in range(1, len(cfg.model.hidden_layers)):
            self.layers.append(GraphConv(cfg.model.hidden_layers[i-1], cfg.model.hidden_layers[i]))

        # Output layer
        self.output_layer = nn.Linear(cfg.model.hidden_layers[-1], num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the GraphConvNet.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.
            edge_weight (torch.Tensor, optional): Edge weights. Default is None.

        Returns:
            torch.Tensor: Log-softmax output.
        """
        for layer in self.layers:
            if edge_weight is not None:
                x = F.relu(layer(x, edge_index, edge_weight=edge_weight))
            else:
                x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding representation of the input.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Embedding representation.
        """
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class GCN_G(nn.Module):
    def __init__(self, num_features: int, num_classes: int, cfg):
        super(GCN_G, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = cfg.model.dropout

        # Input layer
        self.layers.append(GCNConv(num_features, cfg.model.hidden_layers[0]))

        # Hidden layers
        for i in range(1, len(cfg.model.hidden_layers)):
            self.layers.append(GCNConv(cfg.model.hidden_layers[i - 1], cfg.model.hidden_layers[i]))

        # Output layer
        self.output_layer = nn.Linear(cfg.model.hidden_layers[-1], num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_weights: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            if edge_weights is not None:
                x = F.relu(layer(x, edge_index, edge_weight=edge_weights))
            else:
                x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # Replace with global_max_pool if needed
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding representation of the input graph.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch indices for global pooling.

        Returns:
            torch.Tensor: Graph-level embedding representation.
        """
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling for graph-level representation
        x = global_mean_pool(x, batch)  # Replace with global_max_pool if needed
        return x


class ChebNet_G(nn.Module):
    def __init__(self, num_features: int, num_classes: int, cfg):
        super(ChebNet_G, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = cfg.model.dropout

        # Input layer
        self.layers.append(ChebConv(num_features, cfg.model.hidden_layers[0], K=cfg.model.K))

        # Hidden layers
        for i in range(1, len(cfg.model.hidden_layers)):
            self.layers.append(ChebConv(cfg.model.hidden_layers[i - 1], cfg.model.hidden_layers[i], K=cfg.model.K))

        # Output layer
        self.output_layer = nn.Linear(cfg.model.hidden_layers[-1], num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_weights: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            if edge_weights is not None:
                x = F.relu(layer(x, edge_index, edge_weights))
            else:
                x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # Replace with global_max_pool if needed
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding representation of the input graph.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch indices for global pooling.

        Returns:
            torch.Tensor: Graph-level embedding representation.
        """
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling for graph-level representation
        x = global_mean_pool(x, batch)  # Replace with global_max_pool if needed
        return x

class GraphConvNet_G(nn.Module):
    def __init__(self, num_features: int, num_classes: int, cfg):
        super(GraphConvNet_G, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = cfg.model.dropout

        # Input layer
        self.layers.append(GraphConv(num_features, cfg.model.hidden_layers[0]))

        # Hidden layers
        for i in range(1, len(cfg.model.hidden_layers)):
            self.layers.append(GraphConv(cfg.model.hidden_layers[i - 1], cfg.model.hidden_layers[i]))

        # Output layer
        self.output_layer = nn.Linear(cfg.model.hidden_layers[-1], num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_weights: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            if edge_weights is not None:
                x = F.relu(layer(x, edge_index, edge_weight=edge_weights))
            else:
                x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # Replace with global_max_pool if needed
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding representation of the input graph.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch indices for global pooling.

        Returns:
            torch.Tensor: Graph-level embedding representation.
        """
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling for graph-level representation
        x = global_mean_pool(x, batch)  # Replace with global_max_pool if needed
        return x