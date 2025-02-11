import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, subgraph
import sys
from torch_geometric.data import Data
from texttable import Texttable
import torch.optim as optim
from torch import nn, Tensor   
import networkx as nx
import matplotlib.pyplot as plt
import wandb


class TimeOutException(Exception):
    
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        
    
from functools import wraps
import time



def plot_factual_and_counterfactual_graphs(factual_graph: Data, counterfactual_graph: Data, folder: str, pid):
    """
    Plot the factual and counterfactual graphs side by side using networkx and matplotlib.

    Args:
        factual_graph (Data): The factual graph to be plotted.
        counterfactual_graph (Data): The counterfactual graph to be plotted.
    """
    def plot_graph(graph: Data, title: str, ax):
        # Convert edge index to a list of tuples
        edge_list = graph.edge_index.t().tolist()
        
        # Create a networkx graph
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        # Draw the graph
        pos = {i: (graph.x[i, -2].item(), graph.x[i, -1].item()) for i in range(graph.num_nodes)}

        # Draw the graph with node features as labels
        node_labels = {i: f"{graph.x[i, 0]:.0f}, {graph.x[i, 1]:.0f}\n{graph.x[i, 2]:.2f}, {graph.x[i, 3]:.2f}" for i in range(graph.num_nodes)}
        nx.draw(G, pos, labels=node_labels, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=8, font_weight='bold', ax=ax)
            
        ax.set_title(title)
        return pos
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    pos_factual = plot_graph(factual_graph, "Factual", axs[0])
    pos_counterfactual = plot_graph(counterfactual_graph, "Counterfactual", axs[1])
    
    # Determine the common range for x and y axes
    all_x = [pos[0] for pos in pos_factual.values()] + [pos[0] for pos in pos_counterfactual.values()]
    all_y = [pos[1] for pos in pos_factual.values()] + [pos[1] for pos in pos_counterfactual.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    padding = 2
    for ax in axs:
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add padding around each subfigure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
    
    plt.tight_layout()
    plt.savefig(f'data/artifacts/{folder}/factual_and_counterfactual_graphs_{pid}.png')
    plt.close()



def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def get_optimizer(cfg, model):

        if cfg.optimizer.name == "sgd" and cfg.optimizer.n_momentum == 0.0:
            return optim.SGD(model.parameters(), lr=cfg.optimizer.lr)

        elif cfg.optimizer.name == "sgd" and cfg.optimizer.n_momentum != 0.0:
            return optim.SGD(model.parameters(), lr=cfg.optimizer.lr, nesterov=True, momentum=cfg.optimizer.n_momentum)
        
        elif cfg.optimizer.name == "adadelta":
            return optim.Adadelta(model.parameters(), lr=cfg.optimizer.lr)
        
        elif cfg.optimizer.name == "adam":
            return optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
        
        else:
            raise ValueError(f"Optimizer {cfg.optimizer.name} does not exist!")


def get_degree_matrix(adj: torch.Tensor):
    return torch.diag(adj.sum(dim=1))

def print_info(dictionary):

    # Initialize the table
    table = Texttable(max_width=0)
    align_type = ["c"]
    data_type = ["t"]
    cols_valign = ["m"]
    
    table.set_cols_align(align_type*len(dictionary))
    table.set_cols_dtype(data_type*len(dictionary))
    table.set_cols_valign(cols_valign*len(dictionary))
    # Add rows: one row for headers, one for values
    table.add_rows([dictionary.keys(),
                    dictionary.values()])
    
    sys.stdout.write("\033[H\033[J")  # Move cursor to the top and clear screen
    sys.stdout.write(table.draw() + "\n")
    sys.stdout.flush()


def normalize_adj(adj):
    """
    Normalize adjacency matrix using the reparameterization trick from the GCN paper.
    """
    # Add self-loops to the adjacency matrix
    A_tilde = adj + torch.eye(adj.size(0), device=adj.device, dtype=torch.float16)

    # Compute the degree matrix and its inverse square root
    D_tilde = torch.pow(get_degree_matrix(A_tilde), -0.5)
    D_tilde[torch.isinf(D_tilde)] = 0  # Set inf values to 0

    # Compute the normalized adjacency matrix
    norm_adj = D_tilde @ A_tilde @ D_tilde

    return norm_adj


def get_neighbourhood(node_idx: int, edge_index: torch.Tensor, n_hops: int, features: torch.Tensor, labels: torch.Tensor) -> tuple:
    """
    Get the subgraph induced by a node "node_idx" along with all the features and labels.

    Args:
        node_idx (int): Index of the node for which the neighbourhood is to be found.
        edge_index (torch.Tensor): Edge indices of the graph.
        n_hops (int): Number of hops to consider for the neighbourhood.
        features (torch.Tensor): Node features of the graph.
        labels (torch.Tensor): Node labels of the graph.

    Returns:
        tuple: A tuple containing:
            - sub_adj (torch.Tensor): Adjacency matrix of the subgraph.
            - sub_feat (torch.Tensor): Features of the nodes in the subgraph.
            - sub_labels (torch.Tensor): Labels of the nodes in the subgraph.
            - node_dict (dict): Mapping from original node indices to new indices in the subgraph.
    """
    # Get all nodes involved in the k-hop neighbourhood
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)
    
    # Get the relabelled subset of edges
    edge_subset_relabel = subgraph(edge_subset[0], edge_index, relabel_nodes=True)
    
    # Create the adjacency matrix of the subgraph
    #sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    
    # Extract the features of the nodes in the subgraph
    sub_feat = features[edge_subset[0], :]
    
    # Extract the labels of the nodes in the subgraph
    sub_labels = labels[edge_subset[0]]
    
    # Create a mapping from original node indices to new indices in the subgraph
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))
    
    return edge_subset_relabel[0], sub_feat, sub_labels, node_dict



def create_symm_matrix_from_vec(vector, n_rows, device: str = "cpu"):
    matrix = torch.zeros(n_rows, n_rows, device=device)
    idx = torch.tril_indices(n_rows, n_rows, device=device)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def get_S_values(pickled_results, header):
    df_prep = []
    for example in pickled_results:
        if example != []:
            df_prep.append(example[0])
    return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):

    dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)
    
    
def build_factual_graph(mask_index: int, data: Data, n_hops: int, oracle: nn.Module, predicted_labels: Tensor, target_labels: Tensor, device: str = "cuda") -> Data:
    """
    Build a factual graph for a given node in the graph.

    Args:
        mask_index (int): Index of the node for which the factual graph is to be built.
        data (Data): The original graph data.
        n_hops (int): Number of hops to consider for the neighbourhood.
        oracle (nn.Module): The oracle model used to get the embedding representation.
        predicted_labels (Tensor): Predicted labels for the nodes.
        target_labels (Tensor): Target labels for the nodes.
        device (str, optional): Device to use for computations. Defaults to "cuda".

    Returns:
        Data: The factual graph data.
    """
    # Get the neighbourhood of the node
    sub_edge_index, sub_x, sub_labels, node_dict = get_neighbourhood(
        node_idx=int(mask_index),
        edge_index=data.edge_index.cpu(),
        n_hops=n_hops,
        features=data.x.cpu(),
        labels=data.y.cpu()
    )

    # Get the new index of the node in the subgraph
    new_idx = node_dict[int(mask_index)]
    sub_index = list(node_dict.keys())

    # Get the predicted and target labels for the nodes in the subgraph
    sub_y = predicted_labels[sub_index]
    sub_targets = target_labels[sub_index]
    # Move the oracle model to the specified device
    oracle = oracle.to(device)
    # Get the embedding representation from the oracle model
    repr = oracle.get_embedding_repr(sub_x.to(device), sub_edge_index.to(device)).detach()
    embedding_repr = torch.mean(repr, dim=0).to(device)

    # Create the factual graph data
    factual = Data(
        x=sub_x.cpu(),
        edge_index=sub_edge_index.cpu(),
        y=sub_y.cpu(),
        y_ground=sub_labels.cpu(),
        new_idx=new_idx,
        targets=sub_targets.cpu(),
        node_dict=node_dict,
        x_projection=embedding_repr.cpu()
    )

    return factual


def build_counterfactual_graph(x: Tensor, edge_index: Tensor, graph: Data, oracle: nn.Module, output_actual: Tensor, device: str = "cuda") -> Data:
    """
    Constructs a counterfactual graph based on the provided edge index, original graph, and results from an oracle model.
    Args:
        edge_index (Tensor): The edge index tensor representing the edges of the counterfactual graph.
        graph (Data): The original graph data object containing node features and other graph-related information.
        results (dict): A dictionary containing the results from the oracle model, including sparsity and fidelity losses.
        oracle (nn.Module): The oracle neural network model used to obtain embeddings and other necessary computations.
        output_actual (Tensor): The actual output tensor from the oracle model, used to determine the counterfactual labels.
        device (str, optional): The device to perform computations on, default is "cuda".
    Returns:
        Data: A new Data object representing the counterfactual graph with updated attributes.
    """
    
    
    counterfactual = Data(x=x, 
                          edge_index=edge_index, 
                          y=torch.argmax(output_actual, dim=1),
                          sub_index=graph.new_idx,
                          x_projection=torch.mean(oracle.get_embedding_repr(x, edge_index), dim=0))
    
    return counterfactual


def build_counterfactual_graph_gc(x: Tensor, edge_index: Tensor, graph: Data, oracle: nn.Module, output_actual: Tensor, device: str = "cuda") -> Data:
    """
    Constructs a counterfactual graph based on the provided edge index, original graph, and results from an oracle model.
    Args:
        edge_index (Tensor): The edge index tensor representing the edges of the counterfactual graph.
        graph (Data): The original graph data object containing node features and other graph-related information.
        results (dict): A dictionary containing the results from the oracle model, including sparsity and fidelity losses.
        oracle (nn.Module): The oracle neural network model used to obtain embeddings and other necessary computations.
        output_actual (Tensor): The actual output tensor from the oracle model, used to determine the counterfactual labels.
        device (str, optional): The device to perform computations on, default is "cuda".
    Returns:
        Data: A new Data object representing the counterfactual graph with updated attributes.
    """
    
    
    counterfactual = Data(x=x, 
                          edge_index=edge_index, 
                          y=torch.argmax(output_actual, dim=1),
                          x_projection=torch.mean(oracle.get_embedding_repr(x, edge_index, graph.batch), dim=0))
    
    return counterfactual

    


def check_graphs(edge_index) -> bool:
    """
    Check if the input edge index represents an empty or trivial graph.

    A graph is considered empty if there are no edges between any nodes. A graph
    is trivial if it consists of a single node without any self-loops. This function
    checks for these conditions by inspecting the edge index.

    Parameters:
    - edge_index (Tensor): The edge index of a graph.

    Returns:
    - bool: True if the graph is empty or trivial, False otherwise.
    """
    return edge_index.size(1) <= 1




def discretize_tensor(tensor: torch.Tensor) -> torch.Tensor:
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
    discretized_tensor = torch.where(tensor <= -0.5, -1, torch.where(tensor > 0.5, 1, 0))
    return discretized_tensor


def discretize_to_nearest_integer(tensor: torch.Tensor) -> torch.Tensor:
    """
    Discretizes the input tensor to the nearest integer value.

    Args:
        tensor (torch.Tensor): The input float tensor to be discretized.

    Returns:
        torch.Tensor: The discretized tensor containing only integer values.
    """
    return torch.round(tensor)