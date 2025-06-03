"""
Unified Explainer Wrapper for COMBINEX.

This module provides a unified wrapper that handles both node-level and graph-level
explanation tasks with automatic task detection and appropriate processing.
"""

from __future__ import annotations

import copy
import logging
import time
import traceback
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import wandb
from omegaconf import DictConfig
from torch.nn import Module
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from src.abstract.explainer import ExplainerABC
from src.abstract.wrapper import Wrapper
from src.datasets.datainfo import DataInfo
from src.oracles.models.models import GCNLinkPredictor
from src.utils.explainer import get_node_explainer, get_graph_explainer
from ..evaluation.metrics import MetricCalculator
from ..utils.utils import (
    build_factual_graph, 
    check_graphs, 
    plot_factual_and_counterfactual_graphs
)


class UnifiedExplainerWrapper(Wrapper):
    """
    Unified wrapper for both node-level and graph-level explainer tasks.
    
    This wrapper automatically detects the task type based on the input data
    and applies the appropriate explanation methodology.
    
    Args:
        cfg: Configuration object containing model and experiment parameters.
        wandb_run: Weights & Biases run object for logging.
    """
    
    def __init__(self, cfg: DictConfig, wandb_run) -> None:
        super().__init__(cfg=cfg, wandb_run=wandb_run)
        
        # Initialize logger if not already set by parent class
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup reproducibility
        self._setup_reproducibility(cfg.general.seed)
        
        # Task detection
        self.task_type = self._detect_task_type(cfg)
        self.logger.info(f"Detected task type: {self.task_type}")
        
    def _setup_reproducibility(self, seed: int) -> None:
        """Setup reproducibility settings."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(seed)
        
    def _detect_task_type(self, cfg: DictConfig) -> str:
        """
        Detect task type from configuration.
        
        Args:
            cfg: Configuration object.
            
        Returns:
            Task type: 'node' or 'graph'.
        """
        return cfg.task.name.lower()   
        
    def explain(
        self, 
        data: Union[Dataset, Data], 
        datainfo: DataInfo, 
        explainer: str, 
        oracle: Module
    ) -> Optional[Dict]:
        """
        Unified explanation method that handles both node and graph tasks.
        
        Args:
            data: The dataset (Dataset for graph-level, Data for node-level).
            datainfo: Dataset information and metadata.
            explainer: Name of the explainer to use.
            oracle: The trained model to explain.
            
        Returns:
            Dictionary containing explanation results and metrics.
        """
        try:
            if self.task_type == 'node':
                return self._explain_node_level(data, datainfo, explainer, oracle)
            elif self.task_type == 'graph':
                return self._explain_graph_level(data, datainfo, explainer, oracle)
            elif self.task_type == 'link':
                return self._explain_link_level(data, datainfo, explainer, oracle)
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
                
        except Exception as e:
            self.logger.error(f"Error in explanation process: {e}")
            traceback.print_exc()
            return None
            
    def _explain_node_level(
        self, 
        data: Data, 
        datainfo: DataInfo, 
        explainer: str, 
        oracle: Module
    ) -> Optional[Dict]:
        """
        Handle node-level explanation tasks.
        
        Args:
            data: Node-level graph data.
            datainfo: Dataset information.
            explainer: Explainer name.
            oracle: Model to explain.
            
        Returns:
            Dictionary with node-level explanation results.
        """
        self.logger.info(f"Starting node-level explanation with {explainer}")
        
        # Device setup
        device = "cuda" if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu"
        
        # Store current configuration
        self.current_explainer_name = explainer
        self.current_datainfo = datainfo
        
        # Get model predictions
        output = oracle(data.x, data.edge_index).detach()
        predicted_labels = torch.argmax(output, dim=1)
        target_labels = (1 + predicted_labels) % datainfo.num_classes
        
        # Get embedding representation for distribution analysis
        embedding_repr = oracle.get_embedding_repr(data.x, data.edge_index).detach()
        datainfo.distribution_mean_projection = torch.mean(embedding_repr, dim=0).cpu()
        
        # Setup multiprocessing
        mp.set_start_method('spawn', force=True)
        oracle.share_memory()
        
        metric_list = []
        
        try:
            with mp.Manager() as manager:
                queue = manager.Queue(self.cfg.workers)
                results_queue = manager.Queue()
                worker_func = partial(self._node_worker_process, queue, results_queue)
                
                # Create worker processes
                workers = []
                for _ in range(self.cfg.workers):
                    p = mp.Process(target=worker_func)
                    p.start()
                    workers.append(p)
                
                pid = 0
                
                # Process test nodes
                #test_mask_indices = torch.where(data.test_mask)[0] if hasattr(data, 'test_mask') else range(data.x.size(0))
                for mask_index in tqdm(data.test_mask, desc="Processing nodes"):
                    pid += 1
                    
                    # Build factual graph
                    factual_graph = build_factual_graph(
                        mask_index=mask_index,
                        data=data,
                        n_hops=getattr(self.cfg.model, 'n_hops', len(getattr(self.cfg.model, 'hidden_layers', [1])) + 1),
                        oracle=oracle,
                        predicted_labels=predicted_labels,
                        target_labels=target_labels
                    )
                    
                    # Skip invalid graphs
                    if check_graphs(factual_graph.edge_index):
                        continue
                    
                    # Queue task (CPU objects for serialization)
                    args = (
                        oracle.cpu(), factual_graph, explainer, datainfo, 
                        self.cfg, pid, self.wandb_run, device
                    )
                    queue.put(args)
                    
                    if pid >= self.cfg.max_samples:
                        break
                
                # Signal end of tasks
                for _ in range(self.cfg.workers):
                    queue.put(None)
                
                # Wait for completion
                for worker in workers:
                    worker.join()
                
                # Collect results
                while not results_queue.empty():
                    result = results_queue.get()
                    if result is not None:
                        metric_list.append(result)
                
                # Log results
                if metric_list:
                    dataframe = pd.DataFrame.from_dict(metric_list)
                    wandb.log({f"{k}_std": v for k, v in dataframe.std().to_dict().items()})
                    wandb.log({f"{k}_mean": v for k, v in dataframe.mean().to_dict().items()})
                    
                    return {
                        'task_type': 'node',
                        'metrics': dataframe.to_dict('records'),
                        'summary': {
                            'mean': dataframe.mean().to_dict(),
                            'std': dataframe.std().to_dict()
                        }
                    }
                else:
                    self.logger.warning("No valid results collected for node-level explanation")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in node-level explanation: {e}")
            traceback.print_exc()
            return None
            
    def _explain_graph_level(
        self, 
        data: Dataset, 
        datainfo: DataInfo, 
        explainer: str, 
        oracle: Module
    ) -> Optional[Dict]:
        """
        Handle graph-level explanation tasks.
        
        Args:
            data: Graph-level dataset.
            datainfo: Dataset information.
            explainer: Explainer name.
            oracle: Model to explain.
            
        Returns:
            Dictionary with graph-level explanation results.
        """
        self.logger.info(f"Starting graph-level explanation with {explainer}")
        
        # Device setup
        device = "cuda" if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu"
        
        # Store current configuration
        self.current_explainer_name = explainer
        self.current_datainfo = datainfo
        
        self.train_loader = DataLoader(data.dataset, batch_size=8, shuffle=False)
        predicted_labels: torch.Tensor = torch.Tensor([]).to(device)
        embedding_repr: torch.Tensor = torch.Tensor([]).to(device)
        for graphs_batch in self.train_loader:
            graphs_batch = graphs_batch.to(device)
            if "GINENet_G" in str(type(oracle))  or "GAT_G" in str(type(oracle)): 
                input_dict = {"x": graphs_batch.x, "edge_index": graphs_batch.edge_index, "batch": graphs_batch.batch, "edge_attr": graphs_batch.edge_attr}
            else:
                input_dict = {"x": graphs_batch.x, "edge_index": graphs_batch.edge_index, "batch": graphs_batch.batch}
            output = oracle(**input_dict).detach()
            predicted_labels = torch.cat((torch.argmax(output, dim=1), predicted_labels))
            embedding_repr = torch.cat((oracle.get_embedding_repr(**input_dict).detach(), embedding_repr), dim=0)

        datainfo.distribution_mean_projection = embedding_repr.mean(dim=0).cpu()
        target_labels = (1 + predicted_labels) % datainfo.num_classes
        metric_list = []

        self.train_loader = DataLoader(data.dataset[data.test_mask], batch_size=1, shuffle=False)
        # Set the multiprocessing start method and share the oracle model's memory
        mp.set_start_method('spawn', force=True)
        oracle.share_memory()        
        # Setup multiprocessing

        metric_list = []
        
        try:
            # Use a multiprocessing manager to handle queues
            with mp.Manager() as manager:
                queue = manager.Queue(self.cfg.workers)
                results_queue = manager.Queue()
                worker_func = partial(self._graph_worker_process, queue, results_queue)

                # Create and start worker processes
                workers = []
                for _ in range(self.cfg.workers):
                    p = mp.Process(target=worker_func)
                    p.start()
                    workers.append(p)
                    
                pid: int = 0
                
                # Iterate over the test mask indices
                for graph in tqdm(self.train_loader):
                    pid += 1
                    print(f"{pid}/{len(self.train_loader)}")
                    # Build the factual graph for the current mask index
                    
                    arguments = {"x": graph.x.cpu(), 
                                 "edge_index": graph.edge_index.cpu(), 
                                 "batch": graph.batch.cpu(), 
                                 "y": graph.y.cpu(), 
                                 "targets": target_labels[pid].cpu().long(), 
                                 "x_projection": embedding_repr.cpu(), 
                                 "y_ground": predicted_labels[pid].cpu()}
                    
                    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                        arguments["edge_attr"] = graph.edge_attr.cpu()
                    
                    factual = Data(**arguments)

                    # Skip if the graph is invalid
                    if check_graphs(factual.edge_index):
                        continue
                    
                    # Pass everything on CPU because of Queue
                    args = (oracle.cpu(), factual, explainer, datainfo, self.cfg, pid, self.wandb_run, device)
                    queue.put(args)
                    
                    if pid >= self.cfg.max_samples:
                        break
                
                # Signal end of tasks
                for _ in range(self.cfg.workers):
                    queue.put(None)
                
                # Wait for completion
                for worker in workers:
                    worker.join()
                
                # Collect results
                while not results_queue.empty():
                    result = results_queue.get()
                    if result is not None:
                        metric_list.append(result)
                
                
                # Log results
                if metric_list:
                    dataframe = pd.DataFrame.from_dict(metric_list)
                    dataframe = dataframe.fillna(0)  # Fill NaN values with 0
                    wandb.log({f"{k}_std": v for k, v in dataframe.std().to_dict().items()})
                    wandb.log({f"{k}_mean": v for k, v in dataframe.mean().to_dict().items()})
                    
                    return {
                        'task_type': 'graph',
                        'metrics': dataframe.to_dict('records'),
                        'summary': {
                            'mean': dataframe.mean().to_dict(),
                            'std': dataframe.std().to_dict()
                        }
                    }
                else:
                    self.logger.warning("No valid results collected for graph-level explanation")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in graph-level explanation: {e}")
            traceback.print_exc()
            return None
    
    def _explain_link_level(
        self, 
        data: Dataset, 
        datainfo: DataInfo, 
        explainer: str, 
        oracle: GCNLinkPredictor
    ) -> Optional[Dict]:
        """
        Handle graph-level explanation tasks.
        
        Args:
            data: Graph-level dataset.
            datainfo: Dataset information.
            explainer: Explainer name.
            oracle: Model to explain.
            
        Returns:
            Dictionary with graph-level explanation results.
        """
        self.logger.info(f"Starting link-level explanation with {explainer}")
        
        # Device setup
        device = "cuda" if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu"
        
        # Store current configuration
        self.current_explainer_name = explainer
        self.current_datainfo = datainfo
        
       
        predicted_labels: torch.Tensor = torch.where(oracle(x=data.x, edge_index=data.edge_index, predict_edge_index=data.test_pos_edge_index).detach() > 0.5, 1, 0)
        embedding_repr: torch.Tensor = oracle.get_embedding_repr(data.x, data.edge_index).detach().to(device)
        datainfo.distribution_mean_projection = embedding_repr.mean(dim=0).cpu()
        target_labels = (1 + predicted_labels) % datainfo.num_classes
        metric_list = []

        # Set the multiprocessing start method and share the oracle model's memory
        mp.set_start_method('spawn', force=True)
        oracle.share_memory()        
        # Setup multiprocessing

        metric_list = []
        
        try:
            # Use a multiprocessing manager to handle queues
            with mp.Manager() as manager:
                queue = manager.Queue(self.cfg.workers)
                results_queue = manager.Queue()
                worker_func = partial(self._link_worker_process, queue, results_queue)

                # Create and start worker processes
                workers = []
                for _ in range(self.cfg.workers):
                    p = mp.Process(target=worker_func)
                    p.start()
                    workers.append(p)
                    
                pid: int = 0
                
                # Iterate over the test mask indices
                for link in tqdm(range(len(data.train_pos_edge_index[0])), desc="Processing links"):
                    pid += 1
                    
                    edge_index_predict = [data.train_pos_edge_index[0][link], 
                                          data.train_pos_edge_index[1][link]]
                    
                    # print(f"{pid}/{len(self.train_loader)}")
                    # Build the factual graph for the current mask index
                    
                    arguments = {"x": data.x.cpu(), 
                                 "edge_index": data.edge_index.cpu(), 
                                 "y": data.y.cpu(), 
                                 "edge_index_predict": torch.tensor(edge_index_predict).cpu(),
                                 "targets": target_labels[pid].cpu(), 
                                 "x_projection": embedding_repr.cpu(), 
                                 "y_ground": predicted_labels[pid].cpu()}
                    
                    if hasattr(data, "edge_attr") and data.edge_attr is not None:
                        arguments["edge_attr"] = data.edge_attr.cpu()
                    
                    factual = Data(**arguments)

                    # Skip if the graph is invalid
                    if check_graphs(factual.edge_index):
                        continue
                    
                    # Pass everything on CPU because of Queue
                    args = (oracle.cpu(), factual, explainer, datainfo, self.cfg, pid, self.wandb_run, device)
                    queue.put(args)
                    
                    if pid >= self.cfg.max_samples:
                        break
                
                # Signal end of tasks
                for _ in range(self.cfg.workers):
                    queue.put(None)
                
                # Wait for completion
                for worker in workers:
                    worker.join()
                
                # Collect results
                while not results_queue.empty():
                    result = results_queue.get()
                    if result is not None:
                        metric_list.append(result)
                
                
                # Log results
                if metric_list:
                    dataframe = pd.DataFrame.from_dict(metric_list)
                    dataframe = dataframe.fillna(0)  # Fill NaN values with 0
                    wandb.log({f"{k}_std": v for k, v in dataframe.std().to_dict().items()})
                    wandb.log({f"{k}_mean": v for k, v in dataframe.mean().to_dict().items()})
                    
                    return {
                        'task_type': 'graph',
                        'metrics': dataframe.to_dict('records'),
                        'summary': {
                            'mean': dataframe.mean().to_dict(),
                            'std': dataframe.std().to_dict()
                        }
                    }
                else:
                    self.logger.warning("No valid results collected for graph-level explanation")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in graph-level explanation: {e}")
            traceback.print_exc()
            return None
            
    @staticmethod
    def _node_worker_process(queue, results_queue):
        """Worker process for node-level explanations."""
        while True:
            args = queue.get()
            if args is None:
                break
            result = UnifiedExplainerWrapper._process_node_explanation(*args)
            results_queue.put(result)
            
    @staticmethod
    def _graph_worker_process(queue, results_queue):
        """Worker process for graph-level explanations."""
        while True:
            args = queue.get()
            if args is None:
                break
            result = UnifiedExplainerWrapper._process_graph_explanation(*args)
            results_queue.put(result)
    
    @staticmethod
    def _link_worker_process(queue, results_queue):
        """Worker process for link-level explanations."""
        while True:
            args = queue.get()
            if args is None:
                break
            result = UnifiedExplainerWrapper._process_link_explanation(*args)
            results_queue.put(result)
    
    @staticmethod
    def _process_node_explanation(
        oracle: Module, 
        factual_graph: Data, 
        explainer_name: str, 
        datainfo: DataInfo, 
        cfg: DictConfig, 
        pid: int, 
        wandb_run, 
        device: str
    ) -> Optional[Dict]:
        """
        Process a single node explanation task.
        
        Args:
            oracle: Model to explain.
            factual_graph: Factual graph data.
            explainer_name: Name of explainer to use.
            datainfo: Dataset information.
            cfg: Configuration object.
            pid: Process ID for tracking.
            wandb_run: Wandb run object.
            device: Device to use for computation.
            
        Returns:
            Dictionary with explanation metrics or None if failed.
        """
        try:
            # Setup model and explainer
            model = copy.deepcopy(oracle).to(device)
            explainer: ExplainerABC = get_node_explainer(explainer_name)
            explainer = explainer(cfg, datainfo)
            
            # Generate explanation
            start_time = time.time()
            counterfactual = explainer.explain(
                graph=factual_graph.to(device), 
                oracle=model.to(device)
            )
            end_time = time.time()
            
            time_elapsed = end_time - start_time
            
            # Validate counterfactual
            if counterfactual is None or counterfactual.edge_index.shape[1] == 0:
                counterfactual = None
            
            # Generate plots if requested
            if counterfactual is not None and getattr(cfg, 'figure', False):
                plot_factual_and_counterfactual_graphs(
                    factual_graph, counterfactual, 
                    folder=str(wandb_run), pid=pid
                )
            
            # Compute metrics
            calculator = MetricCalculator(task_type='node')
            metrics = calculator.compute_all_metrics(
                factual=factual_graph, 
                counterfactual=counterfactual, 
                mean_projection=getattr(datainfo, 'distribution_mean_projection', None)
            )
            
            # Add time to metrics
            metrics['time_elapsed'] = time_elapsed
            
            print(f"Node explanation completed for pid={pid}")
            return metrics
            
        except Exception as e:
            print(f"Node explanation failed for pid={pid}: {e}")
            traceback.print_exc()
            return None
            
    @staticmethod
    def _process_graph_explanation(
        oracle: Module, 
        graph: Data, 
        explainer_name: str, 
        datainfo: DataInfo, 
        cfg: DictConfig, 
        pid: int, 
        wandb_run, 
        device: str
    ) -> Optional[Dict]:
        """
        Process a single graph explanation task.
        
        Args:
            oracle: Model to explain.
            graph: Graph data to explain.
            explainer_name: Name of explainer to use.
            datainfo: Dataset information.
            cfg: Configuration object.
            pid: Process ID for tracking.
            wandb_run: Wandb run object.
            device: Device to use for computation.
            
        Returns:
            Dictionary with explanation metrics or None if failed.
        """
        try:
            # Setup model and explainer
            model = copy.deepcopy(oracle).to(device)
            explainer: ExplainerABC = get_graph_explainer(explainer_name)
            explainer = explainer(cfg, datainfo)
            
            # Generate explanation
            start_time = time.time()
            explanation = explainer.explain(
                graph=graph.to(device), 
                oracle=model.to(device)
            )
            end_time = time.time()
            
            time_elapsed = end_time - start_time
            
            # Generate plots if requested
            if explanation is not None and getattr(cfg, 'figure', False):
                plot_factual_and_counterfactual_graphs(
                    graph, explanation, 
                    folder=str(wandb_run), pid=pid
                )
            
            # Compute metrics
            calculator = MetricCalculator(task_type='graph')
            metrics = calculator.compute_all_metrics(
                factual=graph, 
                counterfactual=explanation, 
                mean_projection=getattr(datainfo, 'distribution_mean_projection', None)
            )
            
            # Add time to metrics
            metrics['time_elapsed'] = time_elapsed
            
            print(f"Graph explanation completed for pid={pid}")
            return metrics
            
        except Exception as e:
            print(f"Graph explanation failed for pid={pid}: {e}")
            traceback.print_exc()
            return None
            
    @staticmethod
    def _process_link_explanation(
        oracle: Module, 
        graph: Data, 
        explainer_name: str, 
        datainfo: DataInfo, 
        cfg: DictConfig, 
        pid: int, 
        wandb_run, 
        device: str
    ) -> Optional[Dict]:
        """
        Process a single graph explanation task.
        
        Args:
            oracle: Model to explain.
            graph: Graph data to explain.
            explainer_name: Name of explainer to use.
            datainfo: Dataset information.
            cfg: Configuration object.
            pid: Process ID for tracking.
            wandb_run: Wandb run object.
            device: Device to use for computation.
            
        Returns:
            Dictionary with explanation metrics or None if failed.
        """
        try:
            # Setup model and explainer
            model = copy.deepcopy(oracle).to(device)
            explainer: ExplainerABC = get_graph_explainer(explainer_name)
            explainer = explainer(cfg, datainfo)
            
            # Generate explanation
            start_time = time.time()
            explanation = explainer.explain(
                graph=graph.to(device), 
                oracle=model.to(device)
            )
            end_time = time.time()
            
            time_elapsed = end_time - start_time
            
            # Generate plots if requested
            if explanation is not None and getattr(cfg, 'figure', False):
                plot_factual_and_counterfactual_graphs(
                    graph, explanation, 
                    folder=str(wandb_run), pid=pid
                )
            
            # Compute metrics
            calculator = MetricCalculator(task_type='graph')
            metrics = calculator.compute_all_metrics(
                factual=graph, 
                counterfactual=explanation, 
                mean_projection=getattr(datainfo, 'distribution_mean_projection', None)
            )
            
            # Add time to metrics
            metrics['time_elapsed'] = time_elapsed
            
            print(f"Graph explanation completed for pid={pid}")
            return metrics
            
        except Exception as e:
            print(f"Graph explanation failed for pid={pid}: {e}")
            traceback.print_exc()
            return None            

    def get_supported_explainers(self) -> List[str]:
        """
        Get list of supported explainers for current task type.
        
        Returns:
            List of supported explainer names.
        """
        if self.task_type == 'node':
            return [
                'cf-gnnfeatures', 'cf-gnn', 'random-feat', 
                'random', 'cff', 'unr', 'combined'
            ]
        elif self.task_type == 'graph':
            return [
                'combinex', 'gnnexplainer', 'gradcam', 
                'guided_backprop', 'integrated_gradients'
            ]
        else:
            return []
            
    def validate_explainer(self, explainer_name: str) -> bool:
        """
        Validate if explainer is supported for current task type.
        
        Args:
            explainer_name: Name of explainer to validate.
            
        Returns:
            True if supported, False otherwise.
        """
        return explainer_name in self.get_supported_explainers()


# Backward compatibility aliases
NodesExplainerWrapper = UnifiedExplainerWrapper
GraphExplainerWrapper = UnifiedExplainerWrapper


def create_explainer_wrapper(cfg: DictConfig, wandb_run) -> UnifiedExplainerWrapper:
    """
    Factory function to create appropriate explainer wrapper.
    
    Args:
        cfg: Configuration object.
        wandb_run: Wandb run object.
        
    Returns:
        UnifiedExplainerWrapper instance.
    """
    return UnifiedExplainerWrapper(cfg, wandb_run)