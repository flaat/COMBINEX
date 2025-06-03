"""
Training modules for graph neural networks with comprehensive metrics and early stopping.

This module provides trainers for node-level, graph-level, and link prediction tasks
with support for early stopping, multiple metrics, model checkpointing, and logging.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, to_undirected
from torcheval.metrics import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, 
    MulticlassRecall, MulticlassAUROC, BinaryAccuracy, BinaryF1Score,
    BinaryPrecision, BinaryRecall, BinaryAUROC
)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from omegaconf import DictConfig
import wandb
import torch.nn.functional as F


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auroc: Optional[float] = None
    auprc: Optional[float] = None  # Area Under Precision-Recall Curve
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class LinkPredictionData:
    """Container for link prediction data splits."""
    pos_edge_index: torch.Tensor  # Positive edges
    neg_edge_index: torch.Tensor  # Negative edges
    edge_attr: Optional[torch.Tensor] = None  # Edge attributes if available


class EarlyStopping:
    """Early stopping utility with model checkpointing."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            monitor: Metric to monitor ('val_loss', 'val_accuracy', etc.).
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        
    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current metric value.
            epoch: Current epoch number.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'min':
            score_improved = current_score < (self.best_score - self.min_delta)
        else:
            score_improved = current_score > (self.best_score + self.min_delta)
            
        if score_improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
            
        return False


class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(self, cfg: DictConfig, dataset: Union[Dataset, Data], 
                 model: nn.Module, loss_fn: nn.Module):
        """
        Initialize base trainer.
        
        Args:
            cfg: Configuration object.
            dataset: Dataset or data object.
            model: Neural network model.
            loss_fn: Loss function.
        """
        self._setup_reproducibility(cfg.general.seed)
        self.cfg = cfg
        self.dataset = dataset
        self.model = model.to(self._get_device(cfg))
        self.loss_fn = loss_fn
        self.device = self._get_device(cfg)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup metrics
        self.num_classes = self._get_num_classes()
        self.metrics = self._setup_metrics()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=cfg.trainer.get('patience', 20),
            min_delta=cfg.trainer.get('min_delta', 1e-4),
            monitor=cfg.trainer.get('monitor', 'val_loss'),
            mode=cfg.trainer.get('mode', 'min')
        )
        
        # Training history
        self.history = []
        
    def _setup_reproducibility(self, seed: int) -> None:
        """Setup reproducibility settings."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        
    def _get_device(self, cfg: DictConfig) -> str:
        """Get the appropriate device."""
        return "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def _get_num_classes(self) -> int:
        """Get number of classes from dataset."""
        if hasattr(self, '_num_classes_cached'):
            return self._num_classes_cached
            
        if self.cfg.dataset.name == "Facebook":
            self._num_classes_cached = 193
        elif hasattr(self.dataset, 'dataset'):  # Graph-level dataset
            # For graph datasets, get from the data
            unique_labels = self.dataset.dataset.data.y.unique()
            self._num_classes_cached = len(unique_labels)
        else:  # Node-level data
            unique_labels = self.dataset.y.unique()
            self._num_classes_cached = len(unique_labels)
            
        self.logger.info(f"Number of classes detected: {self._num_classes_cached}")
        return self._num_classes_cached
        
    def _setup_metrics(self) -> Dict[str, object]:
        """Setup evaluation metrics with proper device handling."""
        device = self.device
        
        # For link prediction (binary classification)
        if getattr(self.cfg.task, 'name', '').lower() == 'link_prediction':
            return {
                'accuracy': BinaryAccuracy(device=device),
                'f1': BinaryF1Score(device=device),
                'precision': BinaryPrecision(device=device),
                'recall': BinaryRecall(device=device),
                'auroc': BinaryAUROC(device=device)
            }
        
        # For multiclass tasks
        return {
            'accuracy': MulticlassAccuracy(num_classes=self.num_classes, device=device),
            'f1': MulticlassF1Score(num_classes=self.num_classes, average='macro', device=device),
            'precision': MulticlassPrecision(num_classes=self.num_classes, average='macro', device=device),
            'recall': MulticlassRecall(num_classes=self.num_classes, average='macro', device=device),
            'auroc': MulticlassAUROC(num_classes=self.num_classes, average='macro', device=device) if self.num_classes > 2 else None
        }
        
    def _reset_metrics(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            if metric is not None:
                metric.reset()
                
    def _compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                        loss: float, probabilities: Optional[torch.Tensor] = None) -> TrainingMetrics:
        """Compute all metrics with proper error handling."""
        self._reset_metrics()
        
        # Ensure tensors are on the correct device
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)
        if probabilities is not None:
            probabilities = probabilities.to(self.device)
        
        try:
            # Update metrics that expect class predictions
            self.metrics['accuracy'].update(y_pred, y_true)
            self.metrics['f1'].update(y_pred, y_true)
            self.metrics['precision'].update(y_pred, y_true)
            self.metrics['recall'].update(y_pred, y_true)
            
            # For AUROC, we need probabilities, not class predictions
            if self.metrics['auroc'] is not None and probabilities is not None:
                # For binary classification, use the positive class probability
                if probabilities.shape[1] == 2:
                    self.metrics['auroc'].update(probabilities[:, 1], y_true)
                else:
                    self.metrics['auroc'].update(probabilities, y_true)
                    
            # Compute metric values with error handling
            accuracy = self._safe_compute_metric('accuracy')
            f1_score = self._safe_compute_metric('f1')
            precision = self._safe_compute_metric('precision')
            recall = self._safe_compute_metric('recall')
            auroc = self._safe_compute_metric('auroc') if (self.metrics['auroc'] is not None and probabilities is not None) else None
            
            # Compute AUPRC for binary classification tasks
            auprc = None
            if (getattr(self.cfg.task, 'name', '').lower() == 'link_prediction' and 
                probabilities is not None and probabilities.shape[1] == 2):
                try:
                    y_score = probabilities[:, 1].cpu().numpy()
                    y_true_np = y_true.cpu().numpy()
                    auprc = average_precision_score(y_true_np, y_score)
                except Exception as e:
                    self.logger.warning(f"Could not compute AUPRC: {e}")
                    auprc = None
            
        except Exception as e:
            self.logger.warning(f"Error computing metrics: {e}. Using fallback computation.")
            # Fallback to simple accuracy computation
            accuracy = (y_pred == y_true).float().mean().item()
            f1_score = accuracy  # Simple fallback
            precision = accuracy
            recall = accuracy
            auroc = None
            auprc = None
        
        return TrainingMetrics(
            loss=loss,
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            auroc=auroc,
            auprc=auprc
        )
        
    def _safe_compute_metric(self, metric_name: str) -> float:
        """Safely compute a metric with error handling."""
        try:
            if self.metrics[metric_name] is not None:
                result = self.metrics[metric_name].compute()
                if torch.is_tensor(result):
                    return result.item()
                return float(result)
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error computing {metric_name}: {e}")
            return 0.0
        
    def _prepare_model_inputs(self, data: Data) -> Dict[str, torch.Tensor]:
        """Prepare model inputs based on model type."""
        base_inputs = {
            "x": data.x,
            "edge_index": data.edge_index
        }
        
        # Add batch for graph-level tasks
        if hasattr(data, 'batch') and data.batch is not None:
            base_inputs["batch"] = data.batch
            
        # Add edge attributes for specific models
        model_name = self.cfg.model.name
        if model_name in ["GAT", "GINE", "GAT_G", "GINENet_G"] and hasattr(data, 'edge_attr'):
            base_inputs["edge_attr"] = data.edge_attr
            
        return base_inputs
        
    def save_model(self, epoch: Optional[int] = None, is_best: bool = False) -> None:
        """Save model checkpoint."""
        save_dir = Path("data/models")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = f"{self.cfg.dataset.name}_{self.cfg.model.name}"
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch}.pt"
        elif is_best:
            filename = f"{model_name}_best.pt"
        else:
            filename = f"{model_name}_final.pt"
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'epoch': epoch,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, save_dir / filename)
        self.logger.info(f"Model saved: {filename}")
        
    def save_metrics(self) -> None:
        """Save training metrics to CSV."""
        if not self.history:
            return
            
        df = pd.DataFrame(self.history)
        save_dir = Path("data/models")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = (f"{self.cfg.dataset.name}_{self.cfg.model.name}_"
                   f"epochs_{len(self.history)}_metrics.csv")
        df.to_csv(save_dir / filename, index=False)
        self.logger.info(f"Metrics saved: {filename}")
        
    def generate_classification_report(self, y_pred: torch.Tensor, 
                                     y_true: torch.Tensor) -> str:
        """Generate detailed classification report."""
        y_pred_cpu = y_pred.cpu().numpy()
        y_true_cpu = y_true.cpu().numpy()
        
        return classification_report(y_true_cpu, y_pred_cpu)
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler based on configuration."""
        if not self.cfg.trainer.get('use_scheduler', False):
            self.logger.info("No scheduler will be used")
            return None
            
        scheduler_name = self.cfg.trainer.get('scheduler', 'plateau').lower()
        self.logger.info(f"Creating scheduler: {scheduler_name}")
        
        if scheduler_name == 'plateau':
            mode = self.cfg.trainer.get('scheduler_mode', 'min')
            factor = self.cfg.trainer.get('scheduler_factor', 0.5)
            patience = self.cfg.trainer.get('scheduler_patience', 10)
            threshold = self.cfg.trainer.get('scheduler_threshold', 1e-4)
            min_lr = self.cfg.trainer.get('min_lr', 1e-6)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode=mode, 
                factor=factor, 
                patience=patience,
                threshold=threshold,
                min_lr=min_lr,
                verbose=True
            )
            
        elif scheduler_name == 'cosine':
            T_max = self.cfg.trainer.get('scheduler_T_max', self.cfg.trainer.epochs)
            eta_min = self.cfg.trainer.get('eta_min', 1e-6)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=T_max, 
                eta_min=eta_min
            )
            
        elif scheduler_name == 'cosine_restart':
            T_0 = self.cfg.trainer.get('scheduler_T_0', 50)
            T_mult = self.cfg.trainer.get('scheduler_T_mult', 1)
            eta_min = self.cfg.trainer.get('eta_min', 1e-6)
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            
        elif scheduler_name == 'step':
            step_size = self.cfg.trainer.get('scheduler_step_size', 50)
            gamma = self.cfg.trainer.get('scheduler_gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=step_size, 
                gamma=gamma
            )
            
        elif scheduler_name == 'multistep':
            milestones = self.cfg.trainer.get('scheduler_milestones', [50, 100, 150])
            gamma = self.cfg.trainer.get('scheduler_gamma', 0.1)
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
            
        elif scheduler_name == 'exponential':
            gamma = self.cfg.trainer.get('scheduler_gamma', 0.95)
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
            
        elif scheduler_name == 'lambda':
            # Custom lambda function can be defined in config
            lambda_func = self.cfg.trainer.get('scheduler_lambda', lambda epoch: 0.95 ** epoch)
            return optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda_func
            )
            
        elif scheduler_name == 'cyclic':
            base_lr = self.cfg.trainer.get('scheduler_base_lr', self.cfg.trainer.lr * 0.1)
            max_lr = self.cfg.trainer.get('scheduler_max_lr', self.cfg.trainer.lr * 10)
            step_size_up = self.cfg.trainer.get('scheduler_step_size_up', 2000)
            mode = self.cfg.trainer.get('scheduler_cyclic_mode', 'triangular')
            return optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                mode=mode
            )
            
        elif scheduler_name == 'onecycle':
            max_lr = self.cfg.trainer.get('scheduler_max_lr', self.cfg.trainer.lr * 10)
            total_steps = self.cfg.trainer.get('scheduler_total_steps', self.cfg.trainer.epochs)
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=total_steps
            )
            
        elif scheduler_name == 'warmup_cosine':
            # Custom warmup + cosine scheduler
            warmup_epochs = self.cfg.trainer.get('warmup_epochs', 10)
            total_epochs = self.cfg.trainer.epochs
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
                    
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            
        else:
            available_schedulers = [
                'plateau', 'cosine', 'cosine_restart', 'step', 'multistep', 
                'exponential', 'lambda', 'cyclic', 'onecycle', 'warmup_cosine'
            ]
            raise ValueError(
                f"Unsupported scheduler: {scheduler_name}. "
                f"Available schedulers: {available_schedulers}"
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.cfg.trainer.get('optimizer', 'adam').lower()
        lr = self.cfg.trainer.get('lr', 0.01)
        weight_decay = self.cfg.trainer.get('weight_decay', 1e-4)
        
        self.logger.info(f"Creating optimizer: {optimizer_name} with lr={lr}, weight_decay={weight_decay}")
        
        if optimizer_name == 'adam':
            betas = self.cfg.trainer.get('betas', (0.9, 0.999))
            eps = self.cfg.trainer.get('eps', 1e-8)
            return optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
            
        elif optimizer_name == 'adamw':
            betas = self.cfg.trainer.get('betas', (0.9, 0.999))
            eps = self.cfg.trainer.get('eps', 1e-8)
            amsgrad = self.cfg.trainer.get('amsgrad', False)
            return optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                amsgrad=amsgrad
            )
            
        elif optimizer_name == 'sgd':
            momentum = self.cfg.trainer.get('momentum', 0.9)
            dampening = self.cfg.trainer.get('dampening', 0)
            nesterov = self.cfg.trainer.get('nesterov', False)
            return optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov
            )
            
        elif optimizer_name == 'rmsprop':
            alpha = self.cfg.trainer.get('alpha', 0.99)
            eps = self.cfg.trainer.get('eps', 1e-8)
            momentum = self.cfg.trainer.get('momentum', 0)
            centered = self.cfg.trainer.get('centered', False)
            return optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                alpha=alpha,
                eps=eps,
                momentum=momentum,
                centered=centered
            )
            
        elif optimizer_name == 'adagrad':
            lr_decay = self.cfg.trainer.get('lr_decay', 0)
            eps = self.cfg.trainer.get('eps', 1e-10)
            return optim.Adagrad(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                lr_decay=lr_decay,
                eps=eps
            )
            
        elif optimizer_name == 'adadelta':
            rho = self.cfg.trainer.get('rho', 0.9)
            eps = self.cfg.trainer.get('eps', 1e-6)
            return optim.Adadelta(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                rho=rho,
                eps=eps
            )
            
        elif optimizer_name == 'adamax':
            betas = self.cfg.trainer.get('betas', (0.9, 0.999))
            eps = self.cfg.trainer.get('eps', 1e-8)
            return optim.Adamax(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
            
        elif optimizer_name == 'nadam':
            betas = self.cfg.trainer.get('betas', (0.9, 0.999))
            eps = self.cfg.trainer.get('eps', 1e-8)
            momentum_decay = self.cfg.trainer.get('momentum_decay', 4e-3)
            return optim.NAdam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                momentum_decay=momentum_decay
            )
            
        elif optimizer_name == 'radam':
            betas = self.cfg.trainer.get('betas', (0.9, 0.999))
            eps = self.cfg.trainer.get('eps', 1e-8)
            return optim.RAdam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
            
        else:
            available_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 
                                  'adadelta', 'adamax', 'nadam', 'radam']
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                f"Available optimizers: {available_optimizers}"
            )


class LinkPredictionTrainer(BaseTrainer):
    """Trainer for link prediction tasks."""
    
    def __init__(self, cfg: DictConfig, dataset: Data, model: nn.Module, loss_fn: nn.Module):
        super().__init__(cfg, dataset, model, loss_fn)
        
        # Prepare train/val/test edge splits
        self._prepare_edge_splits()
        
    def _get_num_classes(self) -> int:
        """Link prediction is binary classification."""
        return 2
        
    def _prepare_edge_splits(self) -> None:
        """Prepare edge splits for training, validation, and testing."""
        # Get existing edge splits or create them
        if hasattr(self.dataset, 'train_pos_edge_index'):
            # Use existing splits
            self.train_data = LinkPredictionData(
                pos_edge_index=self.dataset.train_pos_edge_index,
                neg_edge_index=getattr(self.dataset, 'train_neg_edge_index', None)
            )
            self.val_data = LinkPredictionData(
                pos_edge_index=self.dataset.val_pos_edge_index,
                neg_edge_index=getattr(self.dataset, 'val_neg_edge_index', None)
            )
            self.test_data = LinkPredictionData(
                pos_edge_index=self.dataset.test_pos_edge_index,
                neg_edge_index=getattr(self.dataset, 'test_neg_edge_index', None)
            )
        else:
            # Create splits manually
            self._create_edge_splits()
            
    def _create_edge_splits(self) -> None:
        """Create edge splits manually."""
        edge_index = self.dataset.edge_index
        num_edges = edge_index.size(1)
        
        # Random split: 80% train, 10% val, 10% test
        perm = torch.randperm(num_edges)
        train_size = int(0.8 * num_edges)
        val_size = int(0.1 * num_edges)
        
        train_edges = edge_index[:, perm[:train_size]]
        val_edges = edge_index[:, perm[train_size:train_size + val_size]]
        test_edges = edge_index[:, perm[train_size + val_size:]]
        
        self.train_data = LinkPredictionData(pos_edge_index=train_edges)
        self.val_data = LinkPredictionData(pos_edge_index=val_edges)
        self.test_data = LinkPredictionData(pos_edge_index=test_edges)
        
    def _sample_negative_edges(self, pos_edge_index: torch.Tensor, 
                             num_nodes: int, num_neg_samples: Optional[int] = None) -> torch.Tensor:
        """Sample negative edges for training."""
        if num_neg_samples is None:
            num_neg_samples = pos_edge_index.size(1)
            
        # Use PyG's negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            method='sparse'
        )
        
        return neg_edge_index
        
    def _predict_links(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict link probabilities."""
        # Prepare inputs for the link prediction model
        inputs = self._prepare_model_inputs(self.dataset)
        
        # Call the model with the required predict_edge_index argument
        edge_scores = self.model(
            x=inputs['x'].to(self.device), 
            edge_index=self.train_data.pos_edge_index.to(self.device),
            predict_edge_index=edge_index.to(self.device)
        )
        
        # Apply sigmoid to get probabilities
        return edge_scores
        
    def _train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        
        # Sample negative edges for this epoch
        pos_edge_index = self.train_data.pos_edge_index.to(self.device)
        neg_edge_index = self._sample_negative_edges(
            pos_edge_index, self.dataset.num_nodes
        ).to(self.device)
        
        # Combine positive and negative edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        
        # Create integer labels (1 for positive edges, 0 for negative edges)
        edge_labels = torch.cat([
            torch.ones(pos_edge_index.size(1), dtype=torch.long),  # Changed to long
            torch.zeros(neg_edge_index.size(1), dtype=torch.long)  # Changed to long
        ]).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        edge_probs = self._predict_links(edge_index)
        
        # Compute loss (need to squeeze edge_probs if it has extra dimensions)
        edge_probs_squeezed = edge_probs.squeeze()
        loss = self.loss_fn(edge_probs_squeezed, edge_labels.float())  # Convert to float for BCE loss
        
        # Backward pass
        loss.backward()
        if self.cfg.trainer.get('clip_grad_norm', 0) > 0:
            clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip_grad_norm)
        self.optimizer.step()
        
        # Compute metrics with proper data types
        y_pred = (edge_probs_squeezed > 0.5).long()  # Convert to long for metrics
        probs = torch.stack([1 - edge_probs_squeezed, edge_probs_squeezed], dim=1)  # Binary probabilities
        
        return self._compute_metrics(y_pred, edge_labels, loss.item(), probs)
        
    def _evaluate(self, data: LinkPredictionData) -> TrainingMetrics:
        """Evaluate on given edge data."""
        self.model.eval()
        
        with torch.no_grad():
            # Sample negative edges for evaluation
            pos_edge_index = data.pos_edge_index.to(self.device)
            
            if data.neg_edge_index is not None:
                neg_edge_index = data.neg_edge_index.to(self.device)
            else:
                neg_edge_index = self._sample_negative_edges(
                    pos_edge_index, self.dataset.num_nodes
                ).to(self.device)
            
            # Combine positive and negative edges
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            
            # Create integer labels (1 for positive edges, 0 for negative edges)
            edge_labels = torch.cat([
                torch.ones(pos_edge_index.size(1), dtype=torch.long),  # Changed to long
                torch.zeros(neg_edge_index.size(1), dtype=torch.long)  # Changed to long
            ]).to(self.device)
            
            # Predict
            edge_probs = self._predict_links(edge_index)
            edge_probs_squeezed = edge_probs.squeeze()
            loss = self.loss_fn(edge_probs_squeezed, edge_labels.float())  # Convert to float for BCE loss
            
            # Compute metrics with proper data types
            y_pred = (edge_probs_squeezed > 0.5).long()  # Convert to long for metrics
            probs = torch.stack([1 - edge_probs_squeezed, edge_probs_squeezed], dim=1)  # Binary probabilities
            
            return self._compute_metrics(y_pred, edge_labels, loss.item(), probs)
            
    def start_training(self) -> Dict[str, List[float]]:
        """Start the training process."""
        self.logger.info(f"Starting link prediction training for {self.cfg.trainer.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.cfg.model.name}")
        self.logger.info(f"Dataset: {self.cfg.dataset.name}")
        
        best_val_metric = float('inf') if 'loss' in self.early_stopping.monitor else 0.0
        
        for epoch in range(self.cfg.trainer.epochs):
            # Training
            train_metrics = self._train_epoch(epoch)
            
            # Validation
            val_metrics = self._evaluate(self.val_data)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.loss)
                else:
                    self.scheduler.step()
                    
            # Log metrics
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics.loss,
                'train_accuracy': train_metrics.accuracy,
                'train_f1': train_metrics.f1_score,
                'val_loss': val_metrics.loss,
                'val_accuracy': val_metrics.accuracy,
                'val_f1': val_metrics.f1_score,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            # Add AUROC if available
            if train_metrics.auroc is not None:
                log_dict['train_auroc'] = train_metrics.auroc
            if val_metrics.auroc is not None:
                log_dict['val_auroc'] = val_metrics.auroc
                
            # Add AUPRC if available
            if train_metrics.auprc is not None:
                log_dict['train_auprc'] = train_metrics.auprc
            if val_metrics.auprc is not None:
                log_dict['val_auprc'] = val_metrics.auprc
                
            self.history.append(log_dict)
            
            # Console logging with None-safe formatting
            train_auroc_str = f"{train_metrics.auroc:.4f}" if train_metrics.auroc is not None else "N/A"
            val_auroc_str = f"{val_metrics.auroc:.4f}" if val_metrics.auroc is not None else "N/A"
            
            self.logger.info(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_metrics.loss:.4f} Acc: {train_metrics.accuracy:.4f} "
                f"AUROC: {train_auroc_str} | "
                f"Val Loss: {val_metrics.loss:.4f} Acc: {val_metrics.accuracy:.4f} "
                f"AUROC: {val_auroc_str}"
            )
            
            # Wandb logging
            if wandb.run is not None:
                wandb.log({f"link_pred_{k}": v for k, v in log_dict.items()})
                
            # Early stopping check
            monitor_metric = getattr(val_metrics, self.early_stopping.monitor.split('_')[-1])
            if self.early_stopping(monitor_metric, epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Save best model
            if ((self.early_stopping.mode == 'min' and monitor_metric < best_val_metric) or
                (self.early_stopping.mode == 'max' and monitor_metric > best_val_metric)):
                best_val_metric = monitor_metric
                self.save_model(epoch, is_best=True)
                
        # Final evaluation
        final_test_metrics = self._evaluate(self.test_data)
        self.logger.info(f"Final test metrics: {final_test_metrics}")
        
        # Generate classification report for link prediction
        self.model.eval()
        with torch.no_grad():
            pos_edge_index = self.test_data.pos_edge_index.to(self.device)
            neg_edge_index = self._sample_negative_edges(
                pos_edge_index, self.dataset.num_nodes
            ).to(self.device)
            
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edge_index.size(1), dtype=torch.long),
                torch.zeros(neg_edge_index.size(1), dtype=torch.long)
            ]).to(self.device)
            
            edge_probs = self._predict_links(edge_index)
            y_pred = (edge_probs.squeeze() > 0.5).long()
            
        report = self.generate_classification_report(y_pred, edge_labels)
        self.logger.info(f"Link Prediction Classification Report:\n{report}")
        
        # Save final artifacts
        self.save_model()
        self.save_metrics()
        
        return {key: [entry[key] for entry in self.history] for key in self.history[0].keys()}


class GraphLevelTrainer(BaseTrainer):
    """Trainer for graph-level prediction tasks."""
    
    def __init__(self, cfg: DictConfig, dataset: Dataset, model: nn.Module, loss_fn: nn.Module):
        super().__init__(cfg, dataset, model, loss_fn)
        
        # Create data loaders
        self.train_loader = DataLoader(
            dataset.dataset[dataset.train_mask],
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.trainer.get('num_workers', 0)
        )
        self.test_loader = DataLoader(
            dataset.dataset[dataset.test_mask],
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            num_workers=cfg.trainer.get('num_workers', 0)
        )
        
        # Validation loader if available
        if hasattr(dataset, 'val_mask'):
            self.val_loader = DataLoader(
                dataset.dataset[dataset.val_mask],
                batch_size=cfg.trainer.batch_size,
                shuffle=False,
                num_workers=cfg.trainer.get('num_workers', 0)
            )
        else:
            self.val_loader = None
            
    def _get_num_classes(self) -> int:
        """Get number of classes for graph-level tasks."""
        if hasattr(self, '_num_classes_cached'):
            return self._num_classes_cached
            
        if self.cfg.dataset.name == "Facebook":
            self._num_classes_cached = 193
        else:
            # Get all unique labels from the entire dataset
            all_labels = []
            for data in self.dataset.dataset:
                if hasattr(data, 'y') and data.y is not None:
                    labels = data.y.flatten() if data.y.dim() > 0 else data.y.unsqueeze(0)
                    all_labels.extend(labels.tolist())
            
            if all_labels:
                unique_labels = set(all_labels)
                self._num_classes_cached = len(unique_labels)
                self.logger.info(f"Found {self._num_classes_cached} unique classes: {sorted(unique_labels)}")
            else:
                # Fallback: try to get from the first data sample
                first_data = self.dataset.dataset[0]
                if hasattr(first_data, 'y'):
                    max_label = int(first_data.y.max().item()) if first_data.y.numel() > 0 else 0
                    self._num_classes_cached = max_label + 1
                else:
                    self._num_classes_cached = 2  # Binary classification fallback
                    
        self.logger.info(f"Number of classes for graph-level task: {self._num_classes_cached}")
        return self._num_classes_cached
        
    def _train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            inputs = self._prepare_model_inputs(data)
            output = self.model(**inputs)
            
            # Compute loss
            targets = data.y.squeeze().long()
            loss = self.loss_fn(output, targets)
            
            # Backward pass
            loss.backward()
            if self.cfg.trainer.get('clip_grad_norm', 0) > 0:
                clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip_grad_norm)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            y_pred = torch.argmax(output, dim=1)
            probs = F.softmax(output, dim=1)  # Convert logits to probabilities
            
            all_preds.append(y_pred)
            all_targets.append(targets)
            all_probs.append(probs)
            
        # Compute epoch metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)
        avg_loss = total_loss / len(self.train_loader)
        
        return self._compute_metrics(all_preds, all_targets, avg_loss, all_probs)
        
    def _evaluate(self, data_loader: DataLoader) -> TrainingMetrics:
        """Evaluate on given data loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                
                inputs = self._prepare_model_inputs(data)
                output = self.model(**inputs)
                
                targets = data.y.squeeze().long()
                loss = self.loss_fn(output, targets)
                
                total_loss += loss.item()
                y_pred = torch.argmax(output, dim=1)
                probs = F.softmax(output, dim=1)  # Convert logits to probabilities
                
                all_preds.append(y_pred)
                all_targets.append(targets)
                all_probs.append(probs)
                
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)
        avg_loss = total_loss / len(data_loader)
        
        return self._compute_metrics(all_preds, all_targets, avg_loss, all_probs)
        
    def start_training(self) -> Dict[str, List[float]]:
        """Start the training process."""
        self.logger.info(f"Starting training for {self.cfg.trainer.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.cfg.model.name}")
        self.logger.info(f"Dataset: {self.cfg.dataset.name}")
        
        best_val_metric = float('inf') if 'loss' in self.early_stopping.monitor else 0.0
        
        for epoch in range(self.cfg.trainer.epochs):
            # Training
            train_metrics = self._train_epoch(epoch)
            
            # Validation/Testing
            if self.val_loader is not None:
                val_metrics = self._evaluate(self.val_loader)
                eval_metrics = val_metrics
                eval_prefix = "val"
            else:
                test_metrics = self._evaluate(self.test_loader)
                eval_metrics = test_metrics
                eval_prefix = "test"
                
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(eval_metrics.loss)
                else:
                    self.scheduler.step()
                    
            # Log metrics
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics.loss,
                'train_accuracy': train_metrics.accuracy,
                'train_f1': train_metrics.f1_score,
                f'{eval_prefix}_loss': eval_metrics.loss,
                f'{eval_prefix}_accuracy': eval_metrics.accuracy,
                f'{eval_prefix}_f1': eval_metrics.f1_score,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            # Add AUROC if available
            if train_metrics.auroc is not None:
                log_dict['train_auroc'] = train_metrics.auroc
                log_dict[f'{eval_prefix}_auroc'] = eval_metrics.auroc
                
            self.history.append(log_dict)
            
            # Console logging
            self.logger.info(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_metrics.loss:.4f} Acc: {train_metrics.accuracy:.4f} | "
                f"{eval_prefix.title()} Loss: {eval_metrics.loss:.4f} Acc: {eval_metrics.accuracy:.4f}"
            )
            
            # Wandb logging
            if wandb.run is not None:
                wandb.log(log_dict)
                
            # Early stopping check
            monitor_metric = getattr(eval_metrics, self.early_stopping.monitor.split('_')[-1])
            if self.early_stopping(monitor_metric, epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Save best model
            if ((self.early_stopping.mode == 'min' and monitor_metric < best_val_metric) or
                (self.early_stopping.mode == 'max' and monitor_metric > best_val_metric)):
                best_val_metric = monitor_metric
                self.save_model(epoch, is_best=True)
                
        # Final evaluation and save
        final_test_metrics = self._evaluate(self.test_loader)
        self.logger.info(f"Final test metrics: {final_test_metrics}")
        
        # Generate classification report
        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for data in self.test_loader:
                data = data.to(self.device)
                inputs = self._prepare_model_inputs(data)
                output = self.model(**inputs)
                y_pred = torch.argmax(output, dim=1)
                all_preds.append(y_pred)
                all_targets.append(data.y.squeeze().long())
                
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
        report = self.generate_classification_report(all_preds, all_targets)
        self.logger.info(f"Classification Report:\n{report}")
        
        # Save final artifacts
        self.save_model()
        self.save_metrics()
        
        return {key: [entry[key] for entry in self.history] for key in self.history[0].keys()}


class NodeLevelTrainer(BaseTrainer):
    """Trainer for node-level prediction tasks."""
    
    def __init__(self, cfg: DictConfig, dataset: Data, model: nn.Module, loss_fn: nn.Module):
        super().__init__(cfg, dataset, model, loss_fn)
        
    def _get_num_classes(self) -> int:
        """Get number of classes for node-level tasks."""
        if hasattr(self, '_num_classes_cached'):
            return self._num_classes_cached
            
        if self.cfg.dataset.name == "Facebook":
            self._num_classes_cached = 193
        else:
            unique_labels = self.dataset.y.unique()
            self._num_classes_cached = len(unique_labels)
            self.logger.info(f"Found {self._num_classes_cached} unique classes: {unique_labels.tolist()}")
            
        return self._num_classes_cached
        
    def _train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        inputs = self._prepare_model_inputs(self.dataset)
        output = self.model(inputs['x'].to(self.device), inputs['edge_index'].to(self.device))
        
        # Compute loss and metrics only on training nodes
        train_mask = self.dataset.train_mask
        targets = self.dataset.y[train_mask].to(self.device)
        loss = self.loss_fn(output[train_mask], targets)
        
        # Backward pass
        loss.backward()
        if self.cfg.trainer.get('clip_grad_norm', 0) > 0:
            clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip_grad_norm)
        self.optimizer.step()
        
        # Compute metrics
        y_pred = torch.argmax(output[train_mask], dim=1)
        probs = F.softmax(output[train_mask], dim=1)  # Convert logits to probabilities
        return self._compute_metrics(y_pred, targets, loss.item(), probs)
        
    def _evaluate(self, mask: torch.Tensor) -> TrainingMetrics:
        """Evaluate on given mask."""
        self.model.eval()
        
        with torch.no_grad():
            inputs = self._prepare_model_inputs(self.dataset)
            output = self.model(inputs['x'].to(self.device), inputs['edge_index'].to(self.device))
            
            targets = self.dataset.y[mask].to(self.device)
            loss = self.loss_fn(output[mask], targets)
            
            y_pred = torch.argmax(output[mask], dim=1)
            probs = F.softmax(output[mask], dim=1)  # Convert logits to probabilities
            return self._compute_metrics(y_pred, targets, loss.item(), probs)
            
    def start_training(self) -> Dict[str, List[float]]:
        """Start the training process."""
        self.logger.info(f"Starting training for {self.cfg.trainer.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.cfg.model.name}")
        self.logger.info(f"Dataset: {self.cfg.dataset.name}")
        
        # Use validation mask if available, otherwise use test mask
        eval_mask = getattr(self.dataset, 'val_mask', self.dataset.test_mask)
        eval_prefix = "val" if hasattr(self.dataset, 'val_mask') else "test"
        
        best_val_metric = float('inf') if 'loss' in self.early_stopping.monitor else 0.0
        
        for epoch in range(self.cfg.trainer.epochs):
            # Training
            train_metrics = self._train_epoch(epoch)
            
            # Evaluation
            eval_metrics = self._evaluate(eval_mask)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(eval_metrics.loss)
                else:
                    self.scheduler.step()
                    
            # Log metrics
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics.loss,
                'train_accuracy': train_metrics.accuracy,
                'train_f1': train_metrics.f1_score,
                f'{eval_prefix}_loss': eval_metrics.loss,
                f'{eval_prefix}_accuracy': eval_metrics.accuracy,
                f'{eval_prefix}_f1': eval_metrics.f1_score,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            # Add AUROC if available
            if train_metrics.auroc is not None:
                log_dict['train_auroc'] = train_metrics.auroc
                log_dict[f'{eval_prefix}_auroc'] = eval_metrics.auroc
                
            self.history.append(log_dict)
            
            # Console logging
            self.logger.info(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_metrics.loss:.4f} Acc: {train_metrics.accuracy:.4f} | "
                f"{eval_prefix.title()} Loss: {eval_metrics.loss:.4f} Acc: {eval_metrics.accuracy:.4f}"
            )
            
            # Wandb logging
            if wandb.run is not None:
                wandb.log({f"oracle_{k}": v for k, v in log_dict.items()})
                
            # Early stopping check
            monitor_metric = getattr(eval_metrics, self.early_stopping.monitor.split('_')[-1])
            if self.early_stopping(monitor_metric, epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Save best model
            if ((self.early_stopping.mode == 'min' and monitor_metric < best_val_metric) or
                (self.early_stopping.mode == 'max' and monitor_metric > best_val_metric)):
                best_val_metric = monitor_metric
                self.save_model(epoch, is_best=True)
                
        # Final evaluation
        final_test_metrics = self._evaluate(self.dataset.test_mask)
        self.logger.info(f"Final test metrics: {final_test_metrics}")
        
        # Generate classification report
        self.model.eval()
        with torch.no_grad():
            inputs = self._prepare_model_inputs(self.dataset)
            output = self.model(inputs['x'].to(self.device), inputs['edge_index'].to(self.device))
            y_pred = torch.argmax(output[self.dataset.test_mask], dim=1)
            y_true = self.dataset.y[self.dataset.test_mask]
            
        report = self.generate_classification_report(y_pred, y_true)
        self.logger.info(f"Classification Report:\n{report}")
        
        # Save final artifacts
        self.save_model()
        self.save_metrics()
        
        return {key: [entry[key] for entry in self.history] for key in self.history[0].keys()}


# Backward compatibility aliases
GraphTrainer = GraphLevelTrainer
Trainer = NodeLevelTrainer


def create_trainer(cfg: DictConfig, dataset: Union[Dataset, Data], 
                  model: nn.Module, loss_fn: nn.Module) -> BaseTrainer:
    """
    Factory function to create appropriate trainer.
    
    Args:
        cfg: Configuration object.
        dataset: Dataset or data object.
        model: Neural network model.
        loss_fn: Loss function.
        
    Returns:
        Appropriate trainer instance.
    """
    # Check if this is link prediction task
    task_name = getattr(cfg.task, 'name', '').lower()
    if task_name == 'link_prediction':
        return LinkPredictionTrainer(cfg, dataset, model, loss_fn)
    
    # Determine if this is graph-level or node-level task
    if hasattr(dataset, 'dataset'):  # Graph-level dataset
        return GraphLevelTrainer(cfg, dataset, model, loss_fn)
    else:  # Node-level data
        return NodeLevelTrainer(cfg, dataset, model, loss_fn)
