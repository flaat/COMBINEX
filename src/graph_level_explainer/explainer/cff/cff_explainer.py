from omegaconf import DictConfig
import torch
from torch_geometric.data import Data
import time
from src.abstract.explainer import Explainer
from src.datasets.dataset import DataInfo
from ...utils.utils import build_counterfactual_graph_gc


class CFFExplainer(Explainer):
    
    def __init__(self, cfg: DictConfig, datainfo: DataInfo) -> None:
        super().__init__(cfg, datainfo)
        self.gam = self.cfg.explainer.gamma
        self.lam = self.cfg.explainer.lam
        self.alp = self.cfg.explainer.alpha
        self.epochs = self.cfg.explainer.epochs
        self.lr = self.cfg.explainer.lr
        self.set_reproducibility()
        
    def name(self):
        return "CFF"

    def explain(self, graph: Data, oracle) -> dict:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        explainer = ExplainModelNodeMulti(
            graph=graph,
            base_model=oracle
        )
        explainer = explainer.to(device)
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.lr, weight_decay=0)
        explainer.train()
        best_loss = torch.inf
        counterfactual = None
        start_time = time.time()
        
        for _ in range(self.epochs):
            
            explainer.zero_grad()
            pred1, pred2 = explainer()
            loss = explainer.loss(pred1.squeeze(0), pred2.squeeze(0), graph.y_ground, self.gam, self.lam, self.alp)
            loss.backward()
            optimizer.step()
            masked_adj = explainer.get_masked_adj()
            out = oracle(x=graph.x, edge_index=masked_adj[0], edge_weights=masked_adj[1], batch=graph.batch)
            edge_index_counterfactual = masked_adj[0][:, masked_adj[1] > 0.5]
            y_new = torch.argmax(out)
            
            if y_new == graph.targets and loss < best_loss:
                
                counterfactual = build_counterfactual_graph_gc(x=graph.x, 
                                                               edge_index=edge_index_counterfactual, 
                                                               graph=graph, 
                                                               oracle=oracle, 
                                                               output_actual=out)
                
                best_loss = loss.item()
            
            if time.time() - start_time > self.cfg.timeout:
                
                return counterfactual

        return counterfactual


                
class ExplainModelNodeMulti(torch.nn.Module):

    def __init__(self, graph, base_model):
        super(ExplainModelNodeMulti, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.graph = graph
        self.base_model = base_model
        self.edge_index = self.graph.edge_index
        self.num_edges = self.edge_index.size(1)

        # Learnable edge mask
        self.edge_mask = torch.nn.Parameter(torch.randn(self.num_edges))
        self.edge_mask.to(device)

    def forward(self):
        masked_adj = self.get_masked_adj()
        S_f = self.base_model(x=self.graph.x, edge_index=masked_adj[0], edge_weights=masked_adj[1], batch=self.graph.batch)  # Counterfactual graph
        S_c = self.base_model(self.graph.x, self.graph.edge_index,  batch=self.graph.batch)  # Original graph
        return S_f, S_c

    def loss(self, S_f, S_c, pred_label, gam, lam, alp):
        relu = torch.nn.ReLU()

        _, sorted_indices = torch.sort(S_f, descending=True)
        S_f_y_k_s = sorted_indices[1]

        _, sorted_indices = torch.sort(S_c, descending=True)
        S_c_y_k_s = sorted_indices[1]

        L_f = relu(gam + S_f[S_f_y_k_s] - S_f[pred_label])
        L_c = relu(gam + S_c[pred_label] - S_c[S_c_y_k_s])

        edge_mask = self.get_edge_mask()
        L1 = torch.linalg.norm(edge_mask, ord=1)

        loss = L1 + lam * (alp * L_f + (1 - alp) * L_c)
        return loss

    def get_masked_adj(self):
        # Apply sigmoid to edge mask to constrain values between 0 and 1
        edge_weights = torch.sigmoid(self.edge_mask)

        # Reconstruct the adjacency matrix with masked edges
        return self.edge_index, edge_weights

    def get_edge_mask(self):
        return torch.sigmoid(self.edge_mask)


