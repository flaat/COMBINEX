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
        
    else:
        raise Exception("Choose a valid dataset!")
    
    

if __name__ == "__main__":
    
    
    data = get_dataset("PROTEINS_G")
    
    print(data)