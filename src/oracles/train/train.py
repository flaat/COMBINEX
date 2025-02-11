import numpy as np
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Dataset
from torcheval.metrics import MulticlassAccuracy
from torch import nn
import wandb
from torch_geometric.loader import DataLoader


class GraphTrainer:

	def __init__(self, cfg: DictConfig, dataset: Dataset, model: nn.Module, loss: nn.Module) -> None:
		# Reproducibility
		torch.manual_seed(cfg.general.seed)
		torch.cuda.manual_seed(cfg.general.seed)
		torch.cuda.manual_seed_all(cfg.general.seed) 
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.autograd.set_detect_anomaly(True)
		np.random.seed(cfg.general.seed)
  
		self.device = "cuda" if torch.cuda.is_available() and cfg.device=="cuda" else "cpu"
		self.cfg = cfg
		self.dataset = dataset
		self.num_classes = self.dataset.dataset.data.y.unique().shape[0] if cfg.dataset.name != "Facebook" else 193
		self.metric = MulticlassAccuracy(num_classes=self.num_classes)
		self.model = model
		self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.trainer.lr)
		self.loss = loss
		self.train_loader = DataLoader(dataset.dataset[dataset.train_mask], batch_size=cfg.trainer.batch_size, shuffle=True)
		self.test_loader = DataLoader(dataset.dataset[dataset.test_mask], batch_size=cfg.trainer.batch_size, shuffle=False)

	def _train(self, epoch: int):
		self.metric.reset()
		self.model.train()
		total_loss = 0
		for data in self.train_loader:
			data = data.to(self.device)
			self.optimizer.zero_grad()
			output = self.model(data.x, data.edge_index, data.batch)
			loss_train = self.loss(output, data.y)
			loss_train.backward()
			clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip)
			self.optimizer.step()
			total_loss += loss_train.item()
			y_pred = torch.argmax(output, dim=1)
			self.metric.update(y_pred, data.y)
		return total_loss / len(self.train_loader), self.metric.compute()

	def _test(self):
		self.metric.reset()
		self.model.eval()
		total_loss = 0
		with torch.no_grad():
			for data in self.test_loader:
				data = data.to(self.device)
				output = self.model(data.x, data.edge_index, data.batch)
				loss_test = self.loss(output, data.y)
				total_loss += loss_test.item()
				y_pred = torch.argmax(output, dim=1)
				self.metric.update(y_pred, data.y)
		return total_loss / len(self.test_loader), self.metric.compute()

	def start_training(self):
		import os
		import pandas as pd
		metrics = pd.DataFrame(columns=["Epoch", "TrainLoss", "TrainAcc", "TestLoss", "TestAcc"])

		for epoch in range(self.cfg.trainer.epochs):
			train_loss, train_accuracy = self._train(epoch=epoch)
			test_loss, test_accuracy = self._test()
			print(f"Epoch: {epoch:4d} Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.4f}")
			wandb.log({
				"train_loss": train_loss,
				"train_accuracy": train_accuracy,
				"test_loss": test_loss,
				"test_accuracy": test_accuracy
			})
			metrics.loc[epoch] = [epoch, train_loss, train_accuracy, test_loss, test_accuracy]

		if not os.path.exists(f"data/models"):
			os.makedirs(f"data/models")

		torch.save(self.model.state_dict(), f"data/models/{self.cfg.dataset.name}_{self.cfg.model.name}_{self.cfg.model.hidden_layers}_epochs_{self.cfg.trainer.epochs}.pt")
		metrics.to_csv(f"data/models/{self.cfg.dataset.name}_{self.cfg.model.name}_{self.cfg.model.hidden_layers}_epochs_{self.cfg.trainer.epochs}_METRICS.csv")


class Trainer:

	def __init__(self, cfg: DictConfig, dataset: Dataset, model: nn.Module, loss: nn.Module) -> None:

		# Reproducibility
		torch.manual_seed(cfg.general.seed)
		torch.cuda.manual_seed(cfg.general.seed)
		torch.cuda.manual_seed_all(cfg.general.seed) 
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.autograd.set_detect_anomaly(True)
		np.random.seed(cfg.general.seed)
		
		self.device = "cuda" if torch.cuda.is_available() and cfg.device=="cuda" else "cpu"
		self.cfg = cfg
		self.dataset = dataset
		self.num_classes = self.dataset.y.unique().shape[0] if cfg.dataset.name != "Facebook" else 193
		self.metric = MulticlassAccuracy(num_classes=self.num_classes)
		self.model = model
		self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.trainer.lr)
		self.loss = loss

	def _train(self, epoch: int):
     
		self.metric.reset()
		self.optimizer.zero_grad()
		output = self.model(self.dataset.x.to(self.device), self.dataset.edge_index.to(self.device))
		loss_train = self.loss(output[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])
		y_pred = torch.argmax(output, dim=1) 
		self.metric.update(y_pred[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])
		loss_train.backward()
		clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip)
		self.optimizer.step()
		return loss_train, self.metric.compute()
		

	def _test(self):

		with torch.no_grad():
			
			self.metric.reset()
			output = self.model(self.dataset.x.to(self.device), self.dataset.edge_index.to(self.device))
			loss_test = self.loss(output[self.dataset.test_mask], self.dataset.y[self.dataset.test_mask])
			y_pred = torch.argmax(output, dim=1) 
			self.metric.update(y_pred[self.dataset.test_mask], self.dataset.y[self.dataset.test_mask])
			return loss_test, self.metric.compute()
	
	def start_training(self):
		import os
		import pandas as pd
		metrics = pd.DataFrame(columns=["Epoch", "TrainLoss", "TrainAcc", "TestLoss", "TestAcc"])

		for epoch in range(self.cfg.trainer.epochs):

			train_loss, train_accuracy = self._train(epoch=epoch)
			test_loss, test_accuracy = self._test()
			print(f"Epoch: {epoch:4d} Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.4f}")
			wandb.log({
				"oracle_train_loss": train_loss.detach().cpu().item(),
				"oracle_train_accuracy": train_accuracy.item(),
				"oracle_test_loss": test_loss.detach().cpu().item(),
				"oracle_test_accuracy": test_accuracy.item()
			})
			metrics.loc[epoch] = [epoch, train_loss.detach().cpu().item(), train_accuracy.item(), test_loss.detach().cpu().item(), test_accuracy.item()]

		if not os.path.exists(f"data/models"):
			os.makedirs(f"data/models")


		torch.save(self.model.state_dict(), f"data/models/{self.cfg.dataset.name}_{self.cfg.model.name}_{self.cfg.model.hidden_layers}_epochs_{self.cfg.trainer.epochs}.pt")
		metrics.to_csv(f"data/models/{self.cfg.dataset.name}_{self.cfg.model.name}_{self.cfg.model.hidden_layers}_epochs_{self.cfg.trainer.epochs}_METRICS.csv")