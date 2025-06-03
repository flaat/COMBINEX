from omegaconf import DictConfig
import torch
from torch import nn
from abc import abstractmethod, ABC

class Perturber(nn.Module, ABC):

    def __init__(self, cfg:DictConfig, model: nn.Module) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.deactivate_model()
        self.set_reproducibility()
        
    def deactivate_model(self):
        """
        Deactivates the model by setting the requires_grad attribute of all model parameters to False.
        This method is used to freeze the model parameters, preventing them from being updated during training.
        """
        
        for param in self.model.parameters():
            param.requires_grad = False

    def set_reproducibility(self)->None:
        """
        Sets the reproducibility settings for PyTorch.

        This method configures the random seed for various PyTorch components to ensure that the results are reproducible.
        It sets the seed for CPU, CUDA, and all CUDA devices. Additionally, it configures the cuDNN backend to be deterministic
        and disables the benchmark mode to avoid non-deterministic algorithms. It also enables anomaly detection for autograd.

        Returns:
            None
        """
        torch.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed_all(self.cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)

    @abstractmethod
    def forward(self):

        pass

    @abstractmethod
    def forward_prediction(self):

        pass


