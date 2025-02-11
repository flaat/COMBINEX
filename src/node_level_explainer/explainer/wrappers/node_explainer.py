from functools import partial
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
from src.abstract.explainer import Explainer
from src.utils.explainer import get_node_explainer
from ....abstract.wrapper import Wrapper
from ...utils.utils import build_factual_graph, check_graphs, plot_factual_and_counterfactual_graphs
from ...evaluation.evaluate import compute_metrics
from src.datasets.dataset import DataInfo
from torch.nn import Module
import torch.multiprocessing as mp
import wandb
from typing import List, Tuple, Dict
import traceback
import copy 
import time   

class NodesExplainerWrapper(Wrapper):
    
    queue = None
    results_queue = None
    
    def __init__(self, cfg: DictConfig, wandb_run) -> None:
        super().__init__(cfg=cfg, wandb_run=wandb_run)

        torch.manual_seed(cfg.general.seed)
        torch.cuda.manual_seed(cfg.general.seed)
        torch.cuda.manual_seed_all(cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(cfg.general.seed)


    def explain(self, data: Dataset, datainfo: DataInfo, explainer: str, oracle: Module) -> dict:
        """
        Explain the node predictions of a graph dataset.

        This method applies the provided explainer to each test instance in the dataset to generate
        counterfactual explanations. It computes and saves the explanation metrics.

        Parameters:
        - data (Dataset): The graph dataset containing features, edges, and test masks.
        - datainfo (DataInfo): Object containing dataset metadata and other relevant information.
        - explainer (Explainer): The explainer algorithm to generate explanations.

        Returns:
        - dict: A dictionary containing the results of the explanation process and metrics.
        """
        print(f"{explainer=}")
        
        # Determine the device to use (GPU if available and configured, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu"
        
        # Set the current explainer name and datainfo
        self.current_explainer_name = explainer
        self.current_datainfo = datainfo
        
        # Get the model's output and predicted labels
        output = oracle(data.x, data.edge_index).detach()
        predicted_labels = torch.argmax(output, dim=1)
        target_labels = (1 + predicted_labels) % datainfo.num_classes
        metric_list = []

        # Get the embedding representation and compute the mean projection
        embedding_repr = oracle.get_embedding_repr(data.x, data.edge_index).detach()
        datainfo.distribution_mean_projection = torch.mean(embedding_repr, dim=0).cpu()

        # Set the multiprocessing start method and share the oracle model's memory
        mp.set_start_method('spawn', force=True)
        oracle.share_memory()
        
        try:
            # Use a multiprocessing manager to handle queues
            with mp.Manager() as manager:
                queue = manager.Queue(self.cfg.workers)
                results_queue = manager.Queue()
                worker_func = partial(self.worker_process, queue, results_queue)

                # Create and start worker processes
                workers = []
                for _ in range(self.cfg.workers):
                    p = mp.Process(target=worker_func)
                    p.start()
                    workers.append(p)
                    
                pid: int = 0
                
                # Iterate over the test mask indices
                for mask_index in tqdm(data.test_mask):
                    pid += 1
                    
                    # Build the factual graph for the current mask index
                    factual_graph = build_factual_graph(mask_index=mask_index,
                                                        data=data,
                                                        n_hops=len(self.cfg.model.hidden_layers) + 1, 
                                                        oracle=oracle, 
                                                        predicted_labels=predicted_labels, 
                                                        target_labels=target_labels)

                    # Skip if the graph is invalid
                    if check_graphs(factual_graph.edge_index):
                        continue
                    
                    # Pass everything on CPU because of Queue
                    args = (oracle.cpu(), factual_graph, explainer, datainfo, self.cfg, pid, self.wandb_run, device)
                    queue.put(args)
                    
                    if pid >= self.cfg.max_samples:
                        break
                
                # Signal the end of tasks
                for _ in range(self.cfg.workers):
                    queue.put(None)
                    
                # Join the worker processes
                for worker in workers:
                    worker.join()

                # Collect the results from the results queue
                while not results_queue.empty():
                    result = results_queue.get()
                    if result is not None:
                        metric_list.append(result)
                
                # Create a DataFrame from the collected metrics and log the mean values
                dataframe = pd.DataFrame.from_dict(metric_list)
                wandb.log(dataframe.std().to_dict())
                wandb.log(dataframe.mean().to_dict())
            
        except Exception as e:
            print(f"{e}")
            traceback.print_exc()
            return None


    @staticmethod
    def worker_process(queue, results_queue):
        """
        Worker process function that continuously retrieves tasks from a queue,
        processes them, and puts the results into another queue.

        Args:
            queue (multiprocessing.Queue): Queue from which tasks (arguments) are retrieved.
            results_queue (multiprocessing.Queue): Queue where processed results are put.

        The function will keep running until it retrieves a `None` value from the queue,
        which signals it to break the loop and terminate.
        """
        while True:
            args = queue.get()
            if args is None:
                break
            result = process(*args)
            results_queue.put(result)

def process(oracle, factual_graph, explainer_name: str, datainfo: DataInfo, cfg, pid: int, wandb_run, device: str = "cuda"):
    try:
        
        model = copy.deepcopy(oracle).to(device)
        explainer: Explainer = get_node_explainer(explainer_name)
        explainer = explainer(cfg, datainfo)

        start_time = time.time()
        counterfactual = explainer.explain(graph=factual_graph.to(device), oracle=model.to(device))
        end_time = time.time()
        
        time_elapsed = end_time - start_time
        if counterfactual is None or counterfactual.edge_index.shape[1] == 0:
            
            counterfactual = None
        
        if counterfactual is not None and cfg.figure:
            plot_factual_and_counterfactual_graphs(factual_graph, counterfactual, folder=wandb_run, pid=pid)
            
            
        metrics = compute_metrics(factual_graph, counterfactual, device=device, data_info=datainfo, time=time_elapsed)
        print(f"Terminated {pid=}")
        return metrics
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        return None


