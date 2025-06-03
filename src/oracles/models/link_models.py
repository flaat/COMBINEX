import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import negative_sampling, train_test_split_edges
from torch_geometric.datasets import Planetoid, CitationFull
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Optional, Tuple
import logging
import os

from src.oracles.models.models import LinkPredictionGNN



class LinkPredictionTrainer:
    """Trainer class for the Link Prediction GNN model."""
    
    def __init__(self, 
                 model: LinkPredictionGNN,
                 device: str = "cuda",
                 lr: float = 0.01,
                 weight_decay: float = 1e-4):
        """
        Initialize the trainer.
        
        Args:
            model: The GNN model
            device: Device to run on
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, data: Data) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get node embeddings
        node_embeddings = self.model(data.x, data.train_pos_edge_index)
        
        # Positive edges
        pos_pred = self.model.predict_links(node_embeddings, data.train_pos_edge_index)
        
        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.train_pos_edge_index.size(1)
        )
        
        # Negative edges
        neg_pred = self.model.predict_links(node_embeddings, neg_edge_index)
        
        # Compute loss
        pos_loss = self.criterion(pos_pred, torch.ones_like(pos_pred))
        neg_loss = self.criterion(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data: Data, split: str = "val") -> Tuple[float, float, float]:
        """Evaluate the model."""
        self.model.eval()
        
        # Get node embeddings
        node_embeddings = self.model(data.x, data.train_pos_edge_index)
        
        # Get the appropriate edge indices
        if split == "val":
            pos_edge_index = data.val_pos_edge_index
            neg_edge_index = data.val_neg_edge_index
        else:  # test
            pos_edge_index = data.test_pos_edge_index
            neg_edge_index = data.test_neg_edge_index
        
        # Predictions
        pos_pred = self.model.predict_links(node_embeddings, pos_edge_index)
        neg_pred = self.model.predict_links(node_embeddings, neg_edge_index)
        
        # Combine predictions and labels
        y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        y_true = torch.cat([
            torch.ones(pos_pred.size(0)),
            torch.zeros(neg_pred.size(0))
        ]).cpu().numpy()
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # Compute metrics
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        accuracy = (y_pred_binary == y_true).mean()
        
        return auc, ap, accuracy
    
    def train(self, 
              data: Data, 
              epochs: int = 500,
              eval_freq: int = 10,
              patience: int = 20) -> dict:
        """
        Train the model with early stopping.
        
        Args:
            data: The graph data
            epochs: Number of training epochs
            eval_freq: Frequency of evaluation
            patience: Early stopping patience
            
        Returns:
            Dictionary with training history
        """
        data = data.to(self.device)
        best_val_auc = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_auc': [], 'val_ap': [], 'val_accuracy': []}
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(data)
            history['train_loss'].append(train_loss)
            
            # Evaluation
            if epoch % eval_freq == 0:
                val_auc, val_ap, val_accuracy = self.evaluate(data, "val")
                history['val_auc'].append(val_auc)
                history['val_ap'].append(val_ap)
                history['val_accuracy'].append(val_accuracy)
                
                self.logger.info(f"Epoch {epoch:03d}: Loss={train_loss:.4f}, "
                               f"Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}, "
                               f"Val Acc={val_accuracy:.4f}")
                
                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_link_prediction_model.pt')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return history

# Example usage and data preparation
def prepare_link_prediction_data(data: Data, val_ratio: float = 0.1, test_ratio: float = 0.2):
    """
    Prepare data for link prediction by splitting edges.
    
    Args:
        data: Original graph data
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        Data object with train/val/test edge splits
    """
    # Remove self-loops and duplicate edges
    from torch_geometric.utils import remove_self_loops, to_undirected
    
    # Convert to undirected and remove self-loops for link prediction
    edge_index = remove_self_loops(data.edge_index)[0]
    edge_index = to_undirected(edge_index)
    
    # Create new data object
    data_clean = Data(x=data.x, edge_index=edge_index)
    
    # Split edges
    data_clean = train_test_split_edges(data_clean, val_ratio=val_ratio, test_ratio=test_ratio)
    return data_clean

def load_real_dataset(dataset_name: str = "Cora", root_dir: str = "./data"):
    """
    Load a real-world dataset for link prediction.
    
    Args:
        dataset_name: Name of the dataset ("Cora", "CiteSeer", "PubMed", "DBLP")
        root_dir: Root directory to store the dataset
        
    Returns:
        Data object
    """
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        # Planetoid datasets (citation networks)
        dataset = Planetoid(root=root_dir, name=dataset_name)
        data = dataset[0]
        
    elif dataset_name == "DBLP":
        # Citation network with more nodes
        dataset = CitationFull(root=root_dir, name="DBLP")
        data = dataset[0]
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    
    return data, dataset.num_classes

# Example training script with real datasets
def main(dataset_name: str = "Cora"):
    """Example training script with real datasets."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load real dataset
    data, num_classes = load_real_dataset(dataset_name)
    
    # Prepare for link prediction
    data = prepare_link_prediction_data(data, val_ratio=0.05, test_ratio=0.1)
    
    print(f"\nAfter edge splitting:")
    print(f"Training edges: {data.train_pos_edge_index.size(1)}")
    print(f"Validation edges: {data.val_pos_edge_index.size(1)}")
    print(f"Test edges: {data.test_pos_edge_index.size(1)}")
    
    # Initialize model with dataset-specific parameters
    model_configs = {
        "Cora": {"hidden_dim": 64, "num_layers": 3, "dropout": 0.5},
        "CiteSeer": {"hidden_dim": 64, "num_layers": 3, "dropout": 0.5},
        "PubMed": {"hidden_dim": 128, "num_layers": 4, "dropout": 0.3},
        "DBLP": {"hidden_dim": 256, "num_layers": 4, "dropout": 0.2}
    }
    
    config = model_configs.get(dataset_name, {"hidden_dim": 64, "num_layers": 3, "dropout": 0.5})
    
    model = LinkPredictionGNN(
        num_features=data.num_features,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        gnn_type="GCN",
        dropout=config["dropout"]
    )
    
    print(f"\nModel configuration:")
    print(f"Hidden dimension: {config['hidden_dim']}")
    print(f"Number of layers: {config['num_layers']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = LinkPredictionTrainer(
        model=model,
        device=device,
        lr=0.01,
        weight_decay=5e-4
    )
    
    # Train the model
    print("\nStarting training...")
    history = trainer.train(data, epochs=500, eval_freq=20, patience=50)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_auc, test_ap, test_accuracy = trainer.evaluate(data, "test")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP: {test_ap:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Best validation AUC: {max(history['val_auc']):.4f}")
    print(f"Best validation AP: {max(history['val_ap']):.4f}")
    print(f"Best validation Accuracy: {max(history['val_accuracy']):.4f}")
    
    return model, history

def benchmark_datasets():
    """Benchmark on multiple datasets."""
    datasets = ["Cora", "CiteSeer", "PubMed"]
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Benchmarking on {dataset_name}")
        print(f"{'='*50}")
        
        try:
            model, history = main(dataset_name)
            
            # Get final test results
            data, _ = load_real_dataset(dataset_name)
            data = prepare_link_prediction_data(data)
            
            trainer = LinkPredictionTrainer(model, device="cuda" if torch.cuda.is_available() else "cpu")
            test_auc, test_ap, test_accuracy = trainer.evaluate(data, "test")
            
            results[dataset_name] = {
                "test_auc": test_auc,
                "test_ap": test_ap,
                "test_accuracy": test_accuracy,
                "best_val_auc": max(history['val_auc']),
                "num_params": sum(p.numel() for p in model.parameters())
            }
            
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
            results[dataset_name] = None
    
    # Print benchmark results
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Test AUC':<10} {'Test AP':<10} {'Test Acc':<10} {'Val AUC':<10} {'Params':<10}")
    print("-" * 80)
    
    for dataset_name, result in results.items():
        if result is not None:
            print(f"{dataset_name:<10} {result['test_auc']:<10.4f} {result['test_ap']:<10.4f} "
                  f"{result['test_accuracy']:<10.4f} {result['best_val_auc']:<10.4f} {result['num_params']:<10}")
        else:
            print(f"{dataset_name:<10} {'FAILED':<10}")

if __name__ == "__main__":
    # Train on a single dataset
    main("Cora")
    
    # Uncomment to benchmark on multiple datasets
    # benchmark_datasets()