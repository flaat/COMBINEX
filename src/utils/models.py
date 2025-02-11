from torch import nn
from src.oracles.models.models import GCN, ChebNet, GCN_G, ChebNet_G, GraphConvNet, GraphConvNet_G
from typing import Union


def get_model(name: str, task: str) -> Union[GCN, ChebNet]:
    print(f"{name=}, {task=}")
    
    if name == "GCN" and task == "Node":
        return GCN
    
    elif name == "CHEB" and task == "Node":
        return ChebNet
    
    elif name == "GraphConv" and task == "Node":
        return GraphConvNet
    
    elif name == "GCN" and task == "Graph":
        return GCN_G
    
    elif name == "CHEB" and task == "Graph":
        return ChebNet_G  
      
    elif name == "GraphConv" and task == "Graph":
        return GraphConvNet_G
    
    else:
        raise ValueError(f"Model not implemented {name}")