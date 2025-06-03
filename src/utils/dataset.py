import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
import torch
import pickle
from torch_geometric.utils import dense_to_sparse, to_undirected
from torch_geometric.transforms import RandomLinkSplit

#TODO aggiungi dataset syntie e aggiungi le maschere e altre cose a tutti i dataset

def get_dataset(dataset_name: str = None, test_size: float = 0.2, task_type: str = None) -> Data:
    """
    Get dataset for different task types.

    Args:
        dataset_name (str, optional): Name of the dataset.
        test_size (float): Test set size ratio.
        task_type (str): Type of task - "node", "graph", or "link".

    Returns:
        Data: PyTorch Geometric Data object.
    """
    
    if dataset_name in ["cora", "pubmed", "citeseer"]:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root="data", name=dataset_name)[0]      
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])]) 
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        if task_type == "link":
            # Create link prediction version
            edge_index = to_undirected(dataset.edge_index)
            
            transform = RandomLinkSplit(
                num_val=0.1,
                num_test=test_size,
                is_undirected=True,
                add_negative_train_samples=False,
                neg_sampling_ratio=1.0,
            )
            
            train_data, val_data, test_data = transform(
                Data(x=dataset.x, edge_index=edge_index, y=dataset.y, num_nodes=dataset.x.size(0))
            )
            
            print(f"Link Prediction Dataset: {dataset_name}")
            print(f"Nodes: {dataset.x.size(0)}")
            print(f"Total edges: {edge_index.size(1)}")
            print(f"Training edges: {train_data.edge_index.size(1)}")
            print(f"Validation edges: {val_data.edge_index.size(1)}")
            print(f"Test edges: {test_data.edge_index.size(1)}")
            
            return Data(
                x=dataset.x,
                edge_index=edge_index,
                y=dataset.y,
                num_nodes=dataset.x.size(0),
                train_pos_edge_index=train_data.edge_index,
                val_pos_edge_index=val_data.edge_index,
                test_pos_edge_index=test_data.edge_index,
                discrete_mask=discrete_mask,
                min_range=min_range,
                max_range=max_range,
                task_type="link",
                dataset_name=dataset_name
            )
        else:
            # Original node classification version
            ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
            train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
            
            return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "karate":
        from torch_geometric.datasets import KarateClub
        
        dataset = KarateClub()[0]
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        if task_type == "link":
            edge_index = to_undirected(dataset.edge_index)
            
            transform = RandomLinkSplit(
                num_val=0.1,
                num_test=test_size,
                is_undirected=True,
                add_negative_train_samples=False,
                neg_sampling_ratio=1.0,
            )
            
            train_data, val_data, test_data = transform(
                Data(x=dataset.x, edge_index=edge_index, y=dataset.y, num_nodes=dataset.x.size(0))
            )
            
            print(f"Karate Club Link Prediction Dataset")
            print(f"Nodes: {dataset.x.size(0)}")
            print(f"Total edges: {edge_index.size(1)}")
            print(f"Training edges: {train_data.edge_index.size(1)}")
            print(f"Validation edges: {val_data.edge_index.size(1)}")
            print(f"Test edges: {test_data.edge_index.size(1)}")
            
            return Data(
                x=dataset.x,
                edge_index=edge_index,
                y=dataset.y,
                num_nodes=dataset.x.size(0),
                train_pos_edge_index=train_data.edge_index,
                val_pos_edge_index=val_data.edge_index,
                test_pos_edge_index=test_data.edge_index,
                discrete_mask=discrete_mask,
                min_range=min_range,
                max_range=max_range,
                task_type="link",
                dataset_name="karate"
            )
        else:
            ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
            train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
            
            return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "twitch":
        from torch_geometric.datasets import Twitch

        dataset = Twitch(root="data", name="EN")[0]
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        if task_type == "link":
            edge_index = to_undirected(dataset.edge_index)
            
            transform = RandomLinkSplit(
                num_val=0.1,
                num_test=test_size,
                is_undirected=True,
                add_negative_train_samples=False,
                neg_sampling_ratio=1.0,
            )
            
            train_data, val_data, test_data = transform(
                Data(x=dataset.x, edge_index=edge_index, y=dataset.y, num_nodes=dataset.x.size(0))
            )
            
            print(f"Twitch Link Prediction Dataset")
            print(f"Nodes: {dataset.x.size(0)}")
            print(f"Total edges: {edge_index.size(1)}")
            print(f"Training edges: {train_data.edge_index.size(1)}")
            print(f"Validation edges: {val_data.edge_index.size(1)}")
            print(f"Test edges: {test_data.edge_index.size(1)}")
            
            return Data(
                x=dataset.x,
                edge_index=edge_index,
                y=dataset.y,
                num_nodes=dataset.x.size(0),
                train_pos_edge_index=train_data.edge_index,
                val_pos_edge_index=val_data.edge_index,
                test_pos_edge_index=test_data.edge_index,
                discrete_mask=discrete_mask,
                min_range=min_range,
                max_range=max_range,
                task_type="link",
                dataset_name="twitch"
            )
        else:
            ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
            train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
            
            return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "actor":
        from torch_geometric.datasets import Actor
        
        dataset = Actor(root="data")[0]
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        if task_type == "link":
            edge_index = to_undirected(dataset.edge_index)
            
            transform = RandomLinkSplit(
                num_val=0.05,  # Smaller validation set for Actor
                num_test=test_size,
                is_undirected=True,
                add_negative_train_samples=False,
                neg_sampling_ratio=1.0,
            )
            
            train_data, val_data, test_data = transform(
                Data(x=dataset.x, edge_index=edge_index, y=dataset.y, num_nodes=dataset.x.size(0))
            )
            
            print(f"Actor Link Prediction Dataset")
            print(f"Nodes: {dataset.x.size(0)}")
            print(f"Total edges: {edge_index.size(1)}")
            print(f"Training edges: {train_data.edge_index.size(1)}")
            print(f"Validation edges: {val_data.edge_index.size(1)}")
            print(f"Test edges: {test_data.edge_index.size(1)}")
            
            return Data(
                x=dataset.x,
                edge_index=edge_index,
                y=dataset.y,
                num_nodes=dataset.x.size(0),
                train_pos_edge_index=train_data.edge_index,
                val_pos_edge_index=val_data.edge_index,
                test_pos_edge_index=test_data.edge_index,
                discrete_mask=discrete_mask,
                min_range=min_range,
                max_range=max_range,
                task_type="link",
                dataset_name="actor"
            )
        else:
            ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
            train_index, test_index = train_test_split(ids, test_size=0.03, random_state=random.randint(0, 100))
            
            return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        from torch_geometric.datasets import WebKB
        
        dataset = WebKB(root="data", name=dataset_name)[0]
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        if task_type == "link":
            edge_index = to_undirected(dataset.edge_index)
            
            transform = RandomLinkSplit(
                num_val=0.1,
                num_test=test_size,
                is_undirected=True,
                add_negative_train_samples=False,
                neg_sampling_ratio=1.0,
            )
            
            train_data, val_data, test_data = transform(
                Data(x=dataset.x, edge_index=edge_index, y=dataset.y, num_nodes=dataset.x.size(0))
            )
            
            print(f"{dataset_name} Link Prediction Dataset")
            print(f"Nodes: {dataset.x.size(0)}")
            print(f"Total edges: {edge_index.size(1)}")
            print(f"Training edges: {train_data.edge_index.size(1)}")
            print(f"Validation edges: {val_data.edge_index.size(1)}")
            print(f"Test edges: {test_data.edge_index.size(1)}")
            
            return Data(
                x=dataset.x,
                edge_index=edge_index,
                y=dataset.y,
                num_nodes=dataset.x.size(0),
                train_pos_edge_index=train_data.edge_index,
                val_pos_edge_index=val_data.edge_index,
                test_pos_edge_index=test_data.edge_index,
                discrete_mask=discrete_mask,
                min_range=min_range,
                max_range=max_range,
                task_type="link",
                dataset_name=dataset_name
            )
        else:
            ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
            train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
            
            return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)   
     
    elif dataset_name in ["Wiki", "BlogCatalog", "Facebook", "PPI"]:
        from torch_geometric.datasets import AttributedGraphDataset

        dataset = AttributedGraphDataset(root="data", name=dataset_name)[0]
        y = dataset.y if dataset_name != "Facebook" else torch.argmax(dataset.y, dim=1)
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        if task_type == "link":
            edge_index = to_undirected(dataset.edge_index)
            
            transform = RandomLinkSplit(
                num_val=0.1,
                num_test=test_size,
                is_undirected=True,
                add_negative_train_samples=False,
                neg_sampling_ratio=1.0,
            )
            
            train_data, val_data, test_data = transform(
                Data(x=dataset.x, edge_index=edge_index, y=y, num_nodes=dataset.x.size(0))
            )
            
            print(f"{dataset_name} Link Prediction Dataset")
            print(f"Nodes: {dataset.x.size(0)}")
            print(f"Total edges: {edge_index.size(1)}")
            print(f"Training edges: {train_data.edge_index.size(1)}")
            print(f"Validation edges: {val_data.edge_index.size(1)}")
            print(f"Test edges: {test_data.edge_index.size(1)}")
            
            return Data(
                x=dataset.x,
                edge_index=edge_index,
                y=y,
                num_nodes=dataset.x.size(0),
                train_pos_edge_index=train_data.edge_index,
                val_pos_edge_index=val_data.edge_index,
                test_pos_edge_index=test_data.edge_index,
                discrete_mask=discrete_mask,
                min_range=min_range,
                max_range=max_range,
                task_type="link",
                dataset_name=dataset_name
            )
        else:
            ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
            train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
            
            return Data(x=dataset.x, edge_index=dataset.edge_index, y=y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)           
        
    elif "syn" in dataset_name:
        with open(f"data/{dataset_name}.pickle","rb") as f:
            data = pickle.load(f)

        adj = torch.Tensor(data["adj"]).squeeze()  
        features = torch.Tensor(data["feat"]).squeeze()
        labels = torch.tensor(data["labels"]).squeeze()
        idx_train = data["train_idx"]
        idx_test = data["test_idx"]
        edge_index = dense_to_sparse(adj)   

        train_index, test_index = train_test_split(idx_train + idx_test, test_size=test_size, random_state=random.randint(0, 100))  
        
        return Data(x=features, edge_index=edge_index[0], y=labels, train_mask=idx_train, test_mask=idx_test)
    
    elif dataset_name == "AIDS":

        class AIDS(InMemoryDataset):
            def __init__(self, root, transform=None, pre_transform=None):
                super(AIDS, self).__init__(root, transform, pre_transform)
                self.data, self.slices = torch.load(self.processed_paths[0])
                self.discrete_mask = torch.Tensor([1, 1, 0, 0])

            @property
            def raw_file_names(self):
                return ["AIDS_A.txt", "AIDS_graph_indicator.txt", "AIDS_graph_labels.txt", "AIDS_node_labels.txt", "AIDS_node_attributes.txt"]

            @property
            def processed_file_names(self):
                return ["data.pt"]

            def download(self):
                pass

            def process(self):
                # Read data into huge `Data` list.
                data_list = []

                # Read files
                edge_index = pd.read_csv(os.path.join(self.raw_dir, "AIDS_A.txt"), sep=",", header=None).values.T
                graph_indicator = pd.read_csv(os.path.join(self.raw_dir, "AIDS_graph_indicator.txt"), sep=",", header=None).values.flatten()
                graph_labels = pd.read_csv(os.path.join(self.raw_dir, "AIDS_graph_labels.txt"), sep=",", header=None).values.flatten()
                node_labels = pd.read_csv(os.path.join(self.raw_dir, "AIDS_node_labels.txt"), sep=",", header=None).values.flatten()
                node_attributes = pd.read_csv(os.path.join(self.raw_dir, "AIDS_node_attributes.txt"), sep=",", header=None).values

                # Process data
                for graph_id in range(1, graph_indicator.max() + 1):
                    node_mask = graph_indicator == graph_id
                    nodes = torch.tensor(node_mask.nonzero()[0].flatten(), dtype=torch.long)
                    x = torch.tensor(node_attributes[node_mask], dtype=torch.float)
                    y = torch.tensor(node_labels[node_mask], dtype=torch.long)

                    edge_mask = (graph_indicator[edge_index[0] - 1] == graph_id) & (graph_indicator[edge_index[1] - 1] == graph_id)
                    edges = torch.tensor(edge_index[:, edge_mask] - 1, dtype=torch.long)

                    data = Data(x=x, edge_index=edges, y=y)
                    data_list.append(data)

                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])

        dataset = AIDS(root="data/AIDS")
        ids = torch.arange(start=0, end=len(dataset.data.x), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]
        print(f"Stats:\nFeatures:{dataset.data.x.shape[1]}\nNodes:{dataset.data.x.shape[0]}\nEdges:{dataset.data.edge_index.shape[1]}\nClasses:{dataset.data.y.max().item()}\n")

        return Data(x=dataset.data.x, edge_index=dataset.data.edge_index, y=dataset.data.y, train_mask=train_index, test_mask=test_index, discrete_mask=dataset.discrete_mask, min_range=min_range, max_range=max_range)

    elif dataset_name == "enzymes":
        class ENZYMES(InMemoryDataset):
            def __init__(self, root, transform=None, pre_transform=None):
                super(ENZYMES, self).__init__(root, transform, pre_transform)
                self.data, self.slices = torch.load(self.processed_paths[0])
                self.discrete_mask = torch.Tensor([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            @property
            def raw_file_names(self):
                return ["ENZYMES_A.txt", "ENZYMES_graph_indicator.txt", "ENZYMES_graph_labels.txt", "ENZYMES_node_labels.txt", "ENZYMES_node_attributes.txt"]

            @property
            def processed_file_names(self):
                return ["data.pt"]

            def download(self):
                pass

            def process(self):
            # Read data into huge `Data` list.
                data_list = []

                # Read files
                edge_index = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_A.txt"), sep=",", header=None).values.T
                graph_indicator = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_graph_indicator.txt"), sep=",", header=None).values.flatten()
                graph_labels = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_graph_labels.txt"), sep=",", header=None).values.flatten()
                node_labels = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_node_labels.txt"), sep=",", header=None).values.flatten()
                node_attributes = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_node_attributes.txt"), sep=",", header=None).values

                # Process data
                for graph_id in range(1, graph_indicator.max() + 1):
                    node_mask = graph_indicator == graph_id
                    nodes = torch.tensor(node_mask.nonzero()[0].flatten(), dtype=torch.long)
                    x = torch.tensor(node_attributes[node_mask], dtype=torch.float)
                    y = torch.tensor(node_labels[node_mask], dtype=torch.long) - 1

                    edge_mask = (graph_indicator[edge_index[0] - 1] == graph_id) & (graph_indicator[edge_index[1] - 1] == graph_id)
                    edges = torch.tensor(edge_index[:, edge_mask] - 1, dtype=torch.long)

                    data = Data(x=x, edge_index=edges, y=y)
                    data_list.append(data)

                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])

        dataset = ENZYMES(root="data/ENZYMES")
        ids = torch.arange(start=0, end=len(dataset.data.x), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]
        
        print(f"Stats:\nFeatures:{dataset.data.x.shape[1]}\nNodes:{dataset.data.x.shape[0]}\nEdges:{dataset.data.edge_index.shape[1]}\nClasses:{dataset.data.y.max().item()}\n")
       
        return Data(x=dataset.data.x, edge_index=dataset.data.edge_index, y=dataset.data.y, train_mask=train_index, test_mask=test_index,  discrete_mask=dataset.discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "protein":
        
        class Proteins(InMemoryDataset):
            def __init__(self, root, transform=None, pre_transform=None):
                super(Proteins, self).__init__(root, transform, pre_transform)
                self.discrete_mask = torch.Tensor([1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

                self.data, self.slices = torch.load(self.processed_paths[0])

            @property
            def raw_file_names(self):
                return ["PROTEINS_full_A.txt", "PROTEINS_full_graph_indicator.txt", "PROTEINS_full_graph_labels.txt", "PROTEINS_full_node_labels.txt", "PROTEINS_full_node_attributes.txt"]

            @property
            def processed_file_names(self):
                return ["data.pt"]

            def download(self):
                pass

            def process(self):
                data_list = []

                edge_index = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_A.txt"), sep=",", header=None).values.T
                graph_indicator = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_graph_indicator.txt"), sep=",", header=None).values.flatten()
                graph_labels = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_graph_labels.txt"), sep=",", header=None).values.flatten()
                node_labels = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_node_labels.txt"), sep=",", header=None).values.flatten()
                node_attributes = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_node_attributes.txt"), sep=",", header=None).values

                for graph_id in range(1, graph_indicator.max() + 1):
                    node_mask = graph_indicator == graph_id
                    nodes = torch.tensor(node_mask.nonzero()[0].flatten(), dtype=torch.long)
                    x = torch.tensor(node_attributes[node_mask], dtype=torch.float)
                    y = torch.tensor(node_labels[node_mask], dtype=torch.long)

                    edge_mask = (graph_indicator[edge_index[0] - 1] == graph_id) & (graph_indicator[edge_index[1] - 1] == graph_id)
                    edges = torch.tensor(edge_index[:, edge_mask] - 1, dtype=torch.long)

                    data = Data(x=x, edge_index=edges, y=y)
                    data_list.append(data)

                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])

        dataset = Proteins(root="data/PROTEINS_full")
        ids = torch.arange(start=0, end=len(dataset.data.x), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.005, random_state=random.randint(0, 100))
        
        print(f"Stats:\nFeatures:{dataset.data.x.shape[1]}\nNodes:{dataset.data.x.shape[0]}\nEdges:{dataset.data.edge_index.shape[1]}\nClasses:{dataset.data.y.max().item()}\n")
        
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]

        return Data(x=dataset.data.x, edge_index=dataset.data.edge_index, y=dataset.data.y, train_mask=train_index, test_mask=test_index,  discrete_mask=dataset.discrete_mask, min_range=min_range, max_range=max_range)

    elif dataset_name == "AIDS-G":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([1, 1, 0, 0] + [1] * 38)
        dataset = TUDataset(root="data/aids", name="AIDS", use_node_attr=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "ENZYMES-G":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([0, 0, 0, 0, 0, 0] + [1] * 15)
        dataset = TUDataset(root="data", name="ENZYMES", use_node_attr=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "PROTEINS-G":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([1, 1, 1, 0, 1, 0, 0, 0, 0] + [1] * 12 + [0] * 8 + [1, 1, 1])
        dataset = TUDataset(root="data/proteins_g", name="PROTEINS_full", use_node_attr=True, force_reload=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)                
    
    elif dataset_name == "COIL-DEL":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([1, 1])
        dataset = TUDataset(root="data", name="COIL-DEL", use_node_attr=True, force_reload=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)                
    
    elif dataset_name == "HIV":
        from torch_geometric.datasets import MoleculeNet

        dataset = MoleculeNet(root="data", name="HIV")
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
                
        # All features are discrete for HIV dataset
        discrete_mask = torch.ones(dataset.data.x.shape[1])
                
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
                
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
                
        train_mask[train_index] = True
        test_mask[test_index] = True
                    
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    elif dataset_name == "FINGERPRINT":
        from torch_geometric.datasets import TUDataset

        # All features are discrete for FINGERPRINT dataset
        discrete_mask = torch.ones(2)
        dataset = TUDataset(root="data", name="Fingerprint", use_node_attr=True, use_edge_attr=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        # Filter out graphs with less than 3 nodes
        valid_indices = []
        for i, graph in enumerate(dataset):
            if graph.x.shape[0] >= 3:  # Check if graph has at least 3 nodes
                valid_indices.append(i)

        if len(valid_indices) < len(dataset):
            print(f"Filtered out {len(dataset) - len(valid_indices)} graphs with less than 3 nodes")
            dataset = dataset.index_select(valid_indices)
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
        
        min_range_edges = torch.min(dataset.data.edge_attr, dim=0)[0]
        max_range_edges = torch.max(dataset.data.edge_attr, dim=0)[0]
        
        
        discrete_edge_attr_mask = torch.zeros(dataset.data.edge_attr.shape[1])
                
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range, min_range_edges=min_range_edges, max_range_edges=max_range_edges, discrete_edge_attr_mask=discrete_edge_attr_mask)
    elif dataset_name == "CUNEIFORM":
        from torch_geometric.datasets import TUDataset

        # All features are discrete for FINGERPRINT dataset
        discrete_mask = torch.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        dataset = TUDataset(root="data", name="Cuneiform", use_node_attr=True, use_edge_attr=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        # Filter out graphs with less than 3 nodes
        valid_indices = []
        for i, graph in enumerate(dataset):
            if graph.x.shape[0] >= 3:  # Check if graph has at least 3 nodes
                valid_indices.append(i)

        if len(valid_indices) < len(dataset):
            print(f"Filtered out {len(dataset) - len(valid_indices)} graphs with less than 3 nodes")
            dataset = dataset.index_select(valid_indices)
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
        
        min_range_edges = torch.min(dataset.data.edge_attr, dim=0)[0]
        max_range_edges = torch.max(dataset.data.edge_attr, dim=0)[0]
        
        
        discrete_edge_attr_mask = torch.Tensor([0, 0, 1, 1])
                
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range, min_range_edges=min_range_edges, max_range_edges=max_range_edges, discrete_edge_attr_mask=discrete_edge_attr_mask)    
    elif dataset_name == "QM9":
        from torch_geometric.datasets import TUDataset
        # We'll use a subset of QM9 features for demonstration purposes
        # QM9 has extensive node and edge features
        dataset = TUDataset(root="data", name="QM9", use_node_attr=True, use_edge_attr=True)
        
        # The node features (atomic numbers, etc.) are all discrete
        discrete_mask = torch.ones(dataset.data.x.shape[1])
        
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]
        
        # Edge features are continuous physical properties
        min_range_edges = torch.min(dataset.data.edge_attr, dim=0)[0]
        max_range_edges = torch.max(dataset.data.edge_attr, dim=0)[0]
        
        # Edge attributes are all continuous (bond properties)
        discrete_edge_attr_mask = torch.zeros(dataset.data.edge_attr.shape[1])
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
        
        print(f"QM9 Dataset: {len(dataset)} molecules with {dataset.data.x.shape[1]} node features, {dataset.data.edge_attr.shape[1]} edge features")
        
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, 
                   discrete_mask=discrete_mask, min_range=min_range, max_range=max_range,
                   min_range_edges=min_range_edges, max_range_edges=max_range_edges,
                   discrete_edge_attr_mask=discrete_edge_attr_mask)
    else:
        # Check if it's a link prediction request for unsupported dataset
        if task_type == "link":
            supported_lp_datasets = [
                "cora", "pubmed", "citeseer", "karate", "twitch", "actor",
                "Cornell", "Texas", "Wisconsin", "Wiki", "BlogCatalog", "Facebook", "PPI"
            ]
            raise ValueError(f"Dataset '{dataset_name}' not supported for link prediction. "
                           f"Supported datasets: {supported_lp_datasets}")
        
        raise Exception("Choose a valid dataset!")


def get_supported_datasets() -> dict:
    """Get dictionary of supported datasets by task type."""
    return {
        "node_classification": [
            "cora", "pubmed", "citeseer", "karate", "twitch", "actor",
            "Cornell", "Texas", "Wisconsin", "Wiki", "BlogCatalog", "Facebook", "PPI"
        ],
        "graph_classification": [
            "AIDS-G", "ENZYMES-G", "PROTEINS-G", "COIL-DEL", "HIV", 
            "FINGERPRINT", "CUNEIFORM", "QM9"
        ],
        "link": [
            "cora", "pubmed", "citeseer", "karate", "twitch", "actor",
            "Cornell", "Texas", "Wisconsin", "Wiki", "BlogCatalog", "Facebook", "PPI"
        ]
    }


if __name__ == "__main__":
    import networkx as nx

    # Test different task types
    print("=" * 50)
    print("Testing Node Classification:")
    data_node = get_dataset("cora", task_type="node")
    print(f"Node data: {data_node}")
    
    print("\n" + "=" * 50)
    print("Testing Link Prediction:")
    data_lp = get_dataset("cora", task_type="link")
    print(f"Link prediction data: {data_lp}")
    print(f"Task type: {getattr(data_lp, 'task_type', 'Not specified')}")
    print(f"Train edges: {data_lp.train_pos_edge_index.size(1)}")
    print(f"Val edges: {data_lp.val_pos_edge_index.size(1)}")
    print(f"Test edges: {data_lp.test_pos_edge_index.size(1)}")
    
    print("\n" + "=" * 50)
    print("Testing Graph Classification:")
    data_graph = get_dataset("AIDS-G", task_type="graph")
    print(f"Graph data: {data_graph}")
    
    # Show supported datasets
    print("\n" + "=" * 50)
    print("Supported datasets:")
    supported = get_supported_datasets()
    for task, datasets in supported.items():
        print(f"{task}: {datasets}")