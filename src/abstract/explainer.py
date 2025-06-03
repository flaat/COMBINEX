from abc import ABC, abstractmethod
from typing import Union
from omegaconf import DictConfig
from torch_geometric.data import Data
import torch
import numpy as np
from src.utils.utils import (
    check_function_factory, 
    get_graph_builder,
    ExplanationContext
)
from src.datasets.datainfo import DataInfo

class ExplainerABC(ABC):

    def __init__(self, cfg: DictConfig, datainfo: DataInfo) -> None:
        
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        self.verbose = cfg.verbose
        self.num_classes = None
        self.datainfo = datainfo
        self.graph_builder = get_graph_builder(cfg.task.name)
        self.check_function = check_function_factory(cfg.task.name)
        self.best_loss = np.inf

    @abstractmethod
    def explain(self, graph: Data, oracle, **kwargs)->dict:

        pass
    

    @abstractmethod
    def name(self):

        pass
    
    
    def set_reproducibility(self):

        # Reproducibility
        torch.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed_all(self.cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(self.cfg.general.seed)	
        
        
    def finalize_explanation(self, ec: ExplanationContext) -> Union[Data, None]:
        """
        Finalizes the explanation by checking the function and building the graph.
        """
        task = self.cfg.task.name.lower()
        counterfactual = None
        
        if task == "graph" or task == "link":
            
            condition_args = {"y_pred_new_actual": ec.y_pred_new_actual,
                    "target": ec.graph.targets,
                    "loss": ec.loss,
                    "best_loss": self.best_loss}
            
            graph_args = {"x": ec.x,
                          "edge_index": ec.edge_index, 
                          "edge_attr": ec.edge_attr,
                          "graph": ec.graph, 
                          "oracle": ec.oracle, 
                          "output_actual": ec.output_actual, 
                          "device": self.device}
            
        elif task == "node" :
            
            condition_args = {"y_pred_new_actual": ec.y_pred_new_actual[ec.graph.new_idx],
                    "target": ec.graph.targets[ec.graph.new_idx],
                    "loss": ec.loss,
                    "best_loss": self.best_loss}
                        
            graph_args = {"x": ec.x,
                          "edge_index": ec.edge_index, 
                          "graph": ec.graph, 
                          "oracle": ec.oracle, 
                          "output_actual": ec.output_actual, 
                          "device": self.device}
        
        
        if self.check_function(**condition_args):
            
            counterfactual = self.graph_builder(**graph_args)
            
            self.best_loss = ec.loss
            
        return counterfactual