from omegaconf import OmegaConf
from torch_geometric.data import Dataset
import torch


class DataInfo:

    def __init__(self, cfg: OmegaConf, data: Dataset) -> None:
        self.device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        self.data = data
        self.distribution_mean = torch.mean(self.data.x, dim=0).cpu() if cfg.task.name == "Node" else torch.mean(self.data.dataset.data.x, dim=0).cpu()
        self._distribution_mean_projection = None
        self.kfold = None
        self.inv_covariance_matrix = None
        self.num_features = self.data.x.shape[1] if cfg.task.name == "Node" else self.data.dataset.data.x.shape[1]
        try:
            self.num_classes = self.data.y.unique().shape[0] if cfg.dataset.name != "Facebook" else 193
        except:
            self.num_classes = self.data.dataset.data.y.unique().shape[0] if cfg.dataset.name != "Facebook" else 193
        self.discrete_mask = self.data.discrete_mask
        self.min_range = self.data.min_range
        self.max_range = self.data.max_range
        del self.data

    def num_classes(self):

        return self.num_classes
    
    def num_features(self):

        return self.num_features
    
    
    def distribution_mean(self):

        return self.distribution_mean 
    
    @property
    def distribution_mean_projection(self):

        return self._distribution_mean_projection
    
    @distribution_mean_projection.setter
    def distribution_mean_projection(self, value):

        self._distribution_mean_projection = value
