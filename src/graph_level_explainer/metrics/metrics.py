import torch
from torch_geometric.data import Data
from scipy.spatial import distance

def sample_distance_from_mean(mean: torch.Tensor, sample: Data) -> float:
    """
    Calculate the Euclidean distance between the mean vector and the mean of the sample's features.

    Parameters:
    mean (torch.Tensor): The mean vector.
    sample (Data): The sample data instance.

    Returns:
    float: The Euclidean distance.
    """
    # Calculate the mean of the sample's features
    sample_mean = torch.mean(sample.x, dim=0)
    # Calculate the Euclidean distance
    distance = torch.sqrt(torch.sum((sample_mean - mean) ** 2))
    return distance.item()

def sample_distance_from_mean_projection(mean: torch.Tensor, sample: Data) -> float:
    """
    Calculate the Euclidean distance between the mean vector and the sample's projected features.

    Parameters:
    mean (torch.Tensor): The mean vector.
    sample (Data): The sample data instance.

    Returns:
    float: The Euclidean distance.
    """
    # Calculate the Euclidean distance
    distance = torch.sqrt(torch.sum((sample.x_projection - mean) ** 2))
    return distance.item()

def factual_counterfactual_distance(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the Euclidean distance between the factual and counterfactual projected features.

    Parameters:
    factual (Data): The factual data instance.
    counterfactual (Data): The counterfactual data instance.

    Returns:
    float: The Euclidean distance.
    """
    # Calculate the Euclidean distance
    distance = torch.sqrt(torch.sum((factual.x_projection - counterfactual.x_projection) ** 2))
    return distance.item()

def fidelity(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the fidelity of the counterfactual explanation.

    Parameters:
    factual (Data): The factual data instance.
    counterfactual (Data): The counterfactual data instance.

    Returns:
    float: The fidelity score.
    """

    phi_G = factual.y
    y = factual.y_ground
    phi_G_i = counterfactual.y
    
    prediction_fidelity = 1 if phi_G == y else 0
    counterfactual_fidelity = 1 if phi_G_i == y else 0
    
    return prediction_fidelity - counterfactual_fidelity


def edge_sparsity(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the edge sparsity between the factual and counterfactual graphs using edge indices.

    Parameters:
    factual (Data): The factual data instance.
    counterfactual (Data): The counterfactual data instance.

    Returns:
    float: The edge sparsity.
    """
    # Get the edge indices
    factual_edges = set(map(tuple, factual.edge_index.t().tolist()))
    counterfactual_edges = set(map(tuple, counterfactual.edge_index.t().tolist()))
    
    # Calculate the number of modified edges
    modified_edges = len(factual_edges.symmetric_difference(counterfactual_edges))
    
    # Calculate the total number of edges in the factual graph
    total_edges = len(factual_edges)
    
    # Calculate the edge sparsity
    sparsity = modified_edges / total_edges
    return sparsity

def node_sparsity(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the node sparsity between the factual and counterfactual graphs.

    Parameters:
    factual (Data): The factual data instance.
    counterfactual (Data): The counterfactual data instance.

    Returns:
    float: The node sparsity.
    """
    # Calculate the number of modified node attributes
    modified_attributes = torch.sum(factual.x != counterfactual.x)
    # Calculate the node sparsity
    sparsity = modified_attributes / factual.x.numel()
    return sparsity.item()

def graph_edit_distance(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the graph edit distance between the factual and counterfactual graphs using edge indices.

    Parameters:
    factual (Data): The factual data instance.
    counterfactual (Data): The counterfactual data instance.

    Returns:
    float: The graph edit distance.
    """
    #TODO: controlla perchè counterfactual.edge_index è vuoto
    
    # Get the edge indices
    factual_edges = set(map(tuple, factual.edge_index.t().tolist()))
    counterfactual_edges = set(map(tuple, counterfactual.edge_index.t().tolist()))
    
    # Calculate the number of modified edges
    modified_edges = len(factual_edges.symmetric_difference(counterfactual_edges))
    
    return modified_edges

def perturbation_distance(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the perturbation distance between the factual and counterfactual node features.

    Parameters:
    factual (Data): The factual data instance.
    counterfactual (Data): The counterfactual data instance.

    Returns:
    float: The perturbation distance.
    """
    # Calculate the perturbation distance
    perturbation_dist = torch.mean((factual.x.long() ^ counterfactual.x.long()).sum(dim=1).float())
    return perturbation_dist.item()

