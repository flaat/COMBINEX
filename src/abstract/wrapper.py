from abc import ABC, abstractmethod
import pickle
from datetime import datetime
from omegaconf import DictConfig
import os
import pandas as pd
import torch

class Wrapper(ABC):

    def __init__(self, cfg: DictConfig, wandb_run) -> None:
        
        self.cfg = cfg
        self.current_explainer_name = None
        self.current_datainfo = None
        self.wandb_run = wandb_run


    @abstractmethod
    def explain(self)->None:

        pass

    def save_data(self, metric_list):

        file_name = f"TECHNIQUE:{self.current_explainer_name}_DATASET:{self.cfg.dataset.name}_MODEL:{self.cfg.model.name}_TASK:{self.cfg.task.name}_SEED:{self.cfg.general.seed}_FOLD:{self.current_datainfo.kfold}_EPOCHS:{self.cfg.trainer.epochs}"
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Define the folder name based on the current timestamp
        folder_name = f"{file_name}_{timestamp}"
        # Define the directory path where you want to create the folder
        # For example, creating the folder in the current working directory
        folder_path = f"{self.cfg.path}/results/{folder_name}"
        print(folder_path, os.getcwd())
        # Check if the folder already exists, and if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(os.getcwd()+folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
        
        dataframe = pd.DataFrame.from_dict(metric_list)
        dataframe.to_csv(f"{os.getcwd()+folder_path}/{file_name}_METRICS.csv")
