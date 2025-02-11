from omegaconf import DictConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ...perturber.pertuber import Perturber
from torch_geometric.data import Data   
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Tuple
from ...utils.utils import discretize_to_nearest_integer
from ...perturber.pertuber import Perturber   
from src.datasets.dataset import DataInfo

class GraphPerturber(Perturber):

    def __init__(self, 
                    cfg: DictConfig, 
                    model: nn.Module, 
                    graph: Data,
                    datainfo: DataInfo,
                    device: str = "cuda") -> None:
        
        super().__init__(cfg=cfg, model=model)
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
        self.beta = 0.5
        self.EP_x = Parameter(torch.ones(len(graph.edge_index[0]), device=self.device))
        self.graph_sample = graph
        self.device = device
        
        # Dataset characteristics
        self.num_classes = datainfo.num_classes
        self.num_nodes = graph.x.shape[0]
        self.num_features = datainfo.num_features
        self.min_range = datainfo.min_range.to(device)
        self.max_range = datainfo.max_range.to(device)
        
        # Explainer characteristics
        self.discrete_features_addition: bool = True
        self.discrete_features_mask: Tensor = datainfo.discrete_mask.to(device)
        self.continous_features_mask: Tensor = 1 - datainfo.discrete_mask.to(device)
        # Model's parameters
        self.P_x = Parameter(torch.zeros(self.num_nodes, self.num_features, device=self.device))
        
        # Graph's components
        self.edge_index = graph.edge_index
        self.x = graph.x         
        
        
    def discretize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Discretizes the input tensor based on the following rules:
        - Values less than or equal to -0.5 are set to -1
        - Values equal to 0.5 are set to 1
        - All other values are set to 0

        Args:
        tensor (torch.Tensor): The input tensor to be discretized.

        Returns:
        torch.Tensor: The discretized tensor.
        """
        discretized_tensor = torch.where(tensor <= 0.5, 0, 1)
        return discretized_tensor.float()
    
    
    def forward(self, V_x) -> Tensor:       
        
        discrete_perturbation = self.discrete_features_mask * torch.clamp(self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x) + V_x, min=self.min_range, max=self.max_range)
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        V_pert = discrete_perturbation + continuous_perturbation

    
        return self.model(V_pert, self.graph_sample.edge_index, F.sigmoid(self.EP_x))

    
    def forward_prediction(self, V_x):
        
        discrete_perturbation = self.discrete_features_mask * discretize_to_nearest_integer(self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x) + V_x)
        discrete_perturbation = torch.clamp(discrete_perturbation, min=self.min_range, max=self.max_range)
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        V_pert = discrete_perturbation + continuous_perturbation

  

        EP_x_discrete = self.discretize_tensor(F.sigmoid(self.EP_x))
        out = self.model(V_pert, self.edge_index, EP_x_discrete)          
        return self.model(V_pert, self.graph_sample.edge_index, EP_x_discrete), out, V_pert, self.EP_x
    
    
    def edge_loss(self, graph):

        # Generate perturbed edge index (with edge weights)
        cf_edge_weights = torch.sigmoid(self.EP_x)  # Learnable edge weights (perturbations)

        # Graph sparsity loss: Penalize large changes in edge weights
        sparsity_loss = torch.sum(torch.abs(cf_edge_weights - 1))  # Penalize deviations from original weights

        cf_edge_weights_discrete = self.discretize_tensor(cf_edge_weights)
        cf_edge_index = graph.edge_index[:, cf_edge_weights_discrete == 1]
        
        return sparsity_loss, cf_edge_index

    def node_loss(self, graph: Data) -> Tuple[Tensor, dict, Tensor]:

        loss_discrete = F.l1_loss(graph.x * self.discrete_features_mask, torch.clamp(self.discrete_features_mask * F.tanh(self.P_x) + graph.x, 0, 1))
        loss_continue = F.mse_loss(graph.x * self.continous_features_mask, self.continous_features_mask * (self.P_x + graph.x))
        
        loss_total = loss_discrete + loss_continue
        
        return loss_total, self.edge_index