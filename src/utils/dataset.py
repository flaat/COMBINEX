import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
import torch
import pickle
from torch_geometric.utils import dense_to_sparse

#TODO aggiungi dataset syntie e aggiungi le maschere e altre cose a tutti i dataset

def get_dataset(dataset_name: str = None, test_size: float = 0.2)->Data:
    """_summary_

    Args:
        dataset_name (str, optional): _description_. Defaults to None.

    Returns:
        Data: _description_
    """
    if dataset_name in ["cora", "pubmed", "citeseer"]:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root="data", name=dataset_name) [0]      
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])]) 
        
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]

        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "karate":
        from torch_geometric.datasets import KarateClub
        
        dataset = KarateClub()[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))

        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "twitch":
        from torch_geometric.datasets import Twitch

        dataset = Twitch(root="data", name="EN")[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]  
        
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "actor":
        from torch_geometric.datasets import Actor
        

        dataset = Actor(root="data")[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        train_index, test_index = train_test_split(ids, test_size=0.03, random_state=random.randint(0, 100))
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        from torch_geometric.datasets import WebKB
        

        dataset = WebKB(root="data", name=dataset_name)[0]  
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)   
     
    elif dataset_name in ["Wiki", "BlogCatalog", "Facebook", "PPI"]:
        from torch_geometric.datasets import AttributedGraphDataset

        dataset = AttributedGraphDataset(root="data", name=dataset_name)[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1)
        y = dataset.y if dataset_name != "Facebook" else torch.argmax(dataset.y, dim=1)
        
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
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
        raise Exception("Choose a valid dataset!")
    
    

if __name__ == "__main__":
    import networkx as nx

    
    data = get_dataset("CUNEIFORM")
    
    print(data)
    print(data.x)
    
    def plot_graph(data, node_size=300, figsize=(10, 10), title="Graph", save_path=None):
        """
        Plot a graph from PyTorch Geometric Data or Dataset using node features as coordinates
        and edge attributes for coloring edges.
        
        Args:
            data: PyTorch Geometric Data or Dataset
            node_size: Size of the nodes
            figsize: Size of the figure
            title: Title of the plot
            save_path: Path to save the figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        
        plt.figure(figsize=figsize)
        
        # Check if it's a single graph or from a dataset
        if hasattr(data, 'dataset'):
            # Get a sample graph from the dataset
            sample = data.dataset[34]
            print(sample)
            print(sample.x)
            print(sample.edge_attr)
            G = nx.Graph()
            
            # Add nodes
            for i in range(sample.x.shape[0]):
                G.add_node(i)
            
            # Add edges with attributes if available
            edge_index = sample.edge_index.t().tolist()
            if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
                for idx, (src, dst) in enumerate(edge_index):
                    G.add_edge(src, dst, attr=sample.edge_attr[idx].tolist())
            else:
                for src, dst in edge_index:
                    G.add_edge(src, dst)
            
            # Store y data for coloring
            if hasattr(sample, 'y'):
                y_data = sample.y
            
            # Use node features as 2D positions (assuming first two features are x,y coords)
            pos = {i: (float(sample.x[i][0]), float(sample.x[i][1])) for i in range(sample.x.shape[0])}
        else:
            # Handle single graph case
            G = nx.Graph()
            for i in range(data.x.shape[0]):
                G.add_node(i)
            
            edge_index = data.edge_index.t().tolist()
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                for idx, (src, dst) in enumerate(edge_index):
                    G.add_edge(src, dst, attr=data.edge_attr[idx].tolist())
            else:
                for src, dst in edge_index:
                    G.add_edge(src, dst)
            
            if hasattr(data, 'y'):
                y_data = data.y
                
            # Use node features as 2D positions
            pos = {i: (float(data.x[i][0]), float(data.x[i][1])) for i in range(data.x.shape[0])}
        
        # Draw the graph
        # Color nodes by class if y is available
        if 'y_data' in locals():
            if y_data.dim() > 1 and y_data.shape[0] == 1:
                # Handle case where y is a single value
                colors = plt.cm.tab10(0)
                nx.draw_networkx_nodes(G, pos, node_color=[colors], node_size=node_size)
            else:
                y_numpy = y_data.numpy().flatten()
                if len(y_numpy) == 1 and len(G.nodes()) > 1:
                    # If there's only one label but multiple nodes, apply it to all
                    y_numpy = np.array([y_numpy[0]] * len(G.nodes()))
                
                # Ensure we have the right number of colors
                if len(y_numpy) != len(G.nodes()):
                    # Default to using a single color if dimensions don't match
                    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size)
                else:
                    # Use a colormap with enough colors for all classes
                    num_classes = len(np.unique(y_numpy))
                    cmap = plt.cm.get_cmap('tab10')
                    nx.draw_networkx_nodes(G, pos, node_color=y_numpy, cmap=cmap, node_size=node_size)
        else:
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size)
        
        # Draw edges with colors based on edge attributes if available
        if any('attr' in d for _, _, d in G.edges(data=True)):
            # Get edge colors from the first attribute in edge_attr
            edge_colors = [d.get('attr', [0])[0] for _, _, d in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, alpha=0.7, width=1.5, edge_color=edge_colors, edge_cmap=plt.cm.coolwarm)
        else:
            nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # Add labels to nodes
        labels = {i: str(i) for i in range(len(G.nodes()))}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    plot_graph(data)