import multiprocessing
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from src.datasets.dataset import DataInfo
from src.graph_level_explainer.explainer.wrappers.graph_explainer import GraphExplainerWrapper
from src.utils.dataset import get_dataset
import random
from src.utils.models import get_model
from src.utils.utils import flatten_dict, merge_hydra_wandb, read_yaml
import os


def log_params(cfg: DictConfig) -> None:
    
    import pandas as pd
    
        
    temp_config = OmegaConf.to_container(cfg)
    config_to_log = flatten_dict(d=temp_config)
    config_to_log = pd.DataFrame([config_to_log])
    config_to_log.astype(str)
    param_table = wandb.Table(dataframe=config_to_log)

    wandb.log({"params": param_table})
    
def set_run_name(cfg, run):
    
    from datetime import datetime
    name = cfg.explainer.name if cfg.explainer.name != "combined" else cfg.explainer.name+f"-{cfg.scheduler.policy}"
    run_name: str = f"{cfg.task.name}_{name}_{cfg.dataset.name}_{cfg.model.name}_{cfg.optimizer.name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run_dir = os.path.join("data", "artifacts", run_name)
    os.makedirs(run_dir, exist_ok=True)
    run.name = run_name
    run.save()


def run_sweep_agent(cfg: DictConfig, sweep_id: str):
    wandb.agent(sweep_id=sweep_id, function=lambda: train(cfg))


def train(cfg):    

    with wandb.init(project=cfg.logger.project, group="experiment_1", mode=cfg.logger.mode) as run:

        from src.node_level_explainer.explainer.wrappers.node_explainer import NodesExplainerWrapper
        from src.oracles.train.train import Trainer, GraphTrainer
        from torch.nn import functional as F
        random.seed(cfg.general.seed)        
        
        merge_hydra_wandb(cfg, wandb.config)
        log_params(cfg)
        set_run_name(cfg, run)

        device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        dataset = get_dataset(cfg.dataset.name, test_size=cfg.test_size)
        datainfo = DataInfo(cfg, dataset)
        wrapper = NodesExplainerWrapper(cfg=cfg, wandb_run=run.name) if cfg.task.name == "Node" else GraphExplainerWrapper(cfg=cfg, wandb_run=run.name)
        oracle = get_model(name=cfg.model.name, task=cfg.task.name)
        oracle = oracle(
            num_features=datainfo.num_features,
            num_classes=datainfo.num_classes,
            cfg=cfg
        )       
        
        dataset = dataset.to(device)
        oracle = oracle.to(device)
        datainfo.kfold = cfg.general.seed
        trainer = Trainer if cfg.task.name == "Node" else GraphTrainer
        trainer = trainer(cfg=cfg, dataset=dataset, model=oracle, loss=F.cross_entropy)
        trainer.start_training()
        oracle = trainer.model
        oracle.eval()
        wrapper.explain(data=dataset, datainfo=datainfo, explainer=cfg.explainer.name, oracle=oracle)       

@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    if cfg.run_mode == 'sweep':

        sweep_config = read_yaml(f'wandb_sweeps_configs/{cfg.logger.config}.yaml')
        sweep_config["name"] = f"{cfg.task.name}_{cfg.dataset.name}_{cfg.model.name}_{cfg.explainer.name}"
        sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.logger.project)
        multiprocessing.set_start_method('spawn', force=True)
        # Number of parallel agents
        num_agents = cfg.num_agents if 'num_agents' in cfg else 1 # Default to 4 agents if not specified
        processes = []
        for _ in range(num_agents):
            p = multiprocessing.Process(target=run_sweep_agent, args=(cfg, sweep_id))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
                    
    elif cfg.run_mode == "run":
        
        train(cfg)
        
    else:
        
        raise ValueError(f"Values for run_mode can be sweep or run, you insert {cfg.run_mode}")
    
if __name__ == "__main__":

    main()
