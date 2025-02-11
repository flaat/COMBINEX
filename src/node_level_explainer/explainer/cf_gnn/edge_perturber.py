from omegaconf import DictConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ...perturber.pertuber import Perturber
from torch_geometric.data import Data   


class EdgePerturber(Perturber):

    def __init__(self, 
                    cfg: DictConfig, 
                    num_classes: int, 
                    model: nn.Module, 
                    graph: Data) -> None:
        
        super().__init__(cfg=cfg, model=model)
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
        self.nclass = num_classes
        self.beta = 0.5
        self.P_x = Parameter(torch.ones(len(graph.edge_index[0]), device=self.device))
        self.graph_sample = graph
          
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
    
    def forward(self) -> Tensor:       
        

        return self.model(self.graph_sample.x, self.graph_sample.edge_index, F.sigmoid(self.P_x))

    
    def forward_prediction(self):

        P_x_discrete = self.discretize_tensor(F.sigmoid(self.P_x))
        
        return self.model(self.graph_sample.x, self.graph_sample.edge_index, P_x_discrete)
    
    def loss(self, graph, output, y_node_non_differentiable):
        """
        Computes the loss for the NodePerturber adapted for counterfactual explanations.

        Args:
        graph: The input graph.
        output: The model output.
        y_node_non_differentiable: The non-differentiable node labels.

        Returns:
        Tuple containing the total loss, a dictionary of individual losses, and the perturbed edge index.
        """
        node_to_explain = graph.new_idx

        # Model's prediction for the node to explain
        y_node_predicted = output[node_to_explain].unsqueeze(0)
        y_node_oracle_original = graph.y[node_to_explain]  # Original ground truth
        y_target = graph.targets[node_to_explain].unsqueeze(0)  # Counterfactual target

        # Calculate if the prediction is already flipped
        pred_same = (y_node_non_differentiable == y_node_oracle_original).float()

        # Generate perturbed edge index (with edge weights)
        cf_edge_weights = torch.sigmoid(self.P_x)  # Learnable edge weights (perturbations)

        # Counterfactual fidelity loss (e.g., cross-entropy)
        fidelity_loss = torch.nn.functional.cross_entropy(y_node_predicted, y_target)

        # Graph sparsity loss: Penalize large changes in edge weights
        sparsity_loss = torch.sum(torch.abs(cf_edge_weights - 1))  # Penalize deviations from original weights

        # Combine losses: fidelity + sparsity regularization
        loss_total: Tensor = pred_same * fidelity_loss + self.beta * sparsity_loss

        cf_edge_weights_discrete = self.discretize_tensor(cf_edge_weights)
        cf_edge_index = graph.edge_index[:, cf_edge_weights_discrete == 1]
        
        return loss_total, cf_edge_index

