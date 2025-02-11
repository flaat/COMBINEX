import copy
import os
import random
import numpy as np
import math
import networkx as nx
from omegaconf import DictConfig
import torch
from src.abstract.explainer import Explainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.graph_level_explainer.utils.utils import build_counterfactual_graph
from .args import parse_args
from .utils import emb_dist_rank, importance
from ...utils.utils import normalize_adj, get_optimizer
import time


class UNRExplainer(Explainer):
    
    
    def __init__(self, cfg: DictConfig, datainfo) -> None:
        super().__init__(cfg, datainfo)
        
        self.set_reproducibility()
        
    
    def explain(self, graph: Data, oracle, **kwargs) -> dict:
        
        counterfactual = None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        arguments = parse_args()

        z = oracle.get_embedding_repr(graph.x, graph.edge_index)
        G = to_networkx(graph, to_undirected=True)

        arguments.expansion_num = max(int(round(np.mean([G.degree[i] for i in list(G.nodes())]))), 3)
        emb_info = emb_dist_rank(z.detach().cpu().numpy(), arguments.neighbors_cnt)

        sub_graph, importance = self.explainer(arguments=arguments, model=oracle, G=G, data=graph, emb_info=emb_info, initial_nd=graph.new_idx, device=device)
                
        edge_index_list = graph.edge_index.t().tolist()
        for edge in sub_graph.edges():
                edge_index_list.remove([edge[0], edge[1]])               
        cf_adj = torch.tensor(edge_index_list, dtype=torch.long).t()
        
        with torch.no_grad():
            output_actual = oracle(graph.x, graph.edge_index)
            output_counterfactual = oracle(graph.x, cf_adj.to(device)).to(device)
        

        if torch.argmax(output_counterfactual, dim=1)[graph.new_idx] != torch.argmax(output_actual, dim=1)[graph.new_idx]:
        
            counterfactual = build_counterfactual_graph(x=graph.x, edge_index=cf_adj.to(device), graph=graph, oracle=oracle, output_actual=output_counterfactual)
        
        return counterfactual
    
    
    def name(self):
        return "UNRExplainer"
        
        
    def explainer(self, arguments, model, G, data, emb_info, initial_nd, device):
        
        bf_top5_idx, bf_dist = emb_info[0] , emb_info[1]
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        
        mcts = MCTS(arguments, G, initial_nd, None)
        mcts.expansion(mcts)
        
        impt_vl = []; impt_sbg = []; num_nodes = []
        
        importance = 0; num_iter = 0; patience = 0; argmax_impt = 0
        n_nodes_khop= khop_sampling(G, initial_nd).number_of_nodes()    
        
        while importance < 1.0: 

            mcts = reset_agent(mcts)
            mcts, subgraph, rw_path = select(arguments, mcts)

            # expansion condition
            if (mcts.C == None) and (mcts.N > 0):
                mcts.expansion(mcts)
                if len(mcts.C) > 0: 
                    subgraph.add_edge(mcts.state, mcts.C[0].state)
                    mcts = mcts.C[0]
                    rw_path.append(0)
                else:
                    if subgraph.number_of_nodes() ==1:
                        break
            else:
                pass

            importance = simulate(arguments, subgraph, initial_nd, model, G, bf_top5_idx, bf_dist, x, edge_index)

            n_nodes = subgraph.number_of_nodes()
            num_nodes.append(n_nodes)        
            backprop(mcts, rw_path, importance)

            impt_vl.append(importance)
            impt_sbg.append(subgraph)
            num_iter += 1

            print('initial node: ', initial_nd, ' | num try: ', num_iter,' | # of nodes: ', n_nodes, ' | importance: ', importance)
            
            
            if n_nodes ==1:
                break    
            elif n_nodes_khop==2:
                break
            elif (n_nodes_khop==3)and(num_iter > 100):
                break   

            if importance > argmax_impt:
                argmax_impt = importance
                patience = 0
            else:
                patience += 1

                if (patience > 10) and (num_iter > 500):
                    break
                else:
                    pass
                
        if n_nodes==1:
            return subgraph, 0
        else:
            max_score = max(impt_vl)    
            max_lst = np.where(np.array(impt_vl) == max_score)[0]
            min_nodes = min([v for i,v in enumerate(num_nodes) if i in max_lst])
            fn_idx = [i for i,v in enumerate(num_nodes) if v ==min_nodes and i in max_lst][0]
            fn_sbg = impt_sbg[fn_idx]
            fn_score = impt_vl[fn_idx]

            return fn_sbg, fn_score
        
        





class MCTS:
    def __init__(self, args, G, nd, parent):
        self.args = args
        self.G = G
        self.state = nd
        self.V = 0  
        self.N = 0 
        self.Vi = 0
        self.parent = parent
        self.C = None
        
    def expansion(self, parent):
        n_lst = [n for n in self.G.neighbors(self.state)]
        n_lst_idx = np.random.choice(len(n_lst), min(self.args.expansion_num, len(n_lst)), replace=False)
        n_lst = [n_lst[idx] for idx in n_lst_idx]            
        self.C = {i: MCTS(self.args, self.G, v, parent) for i, v in enumerate(n_lst)}

def UCB1( vi, N, n, c1):
    if n > 0:
        return vi + c1*(math.sqrt(math.log(N)/n))
    else:
        return math.inf
    
def select(args, mcts):
    
    N = mcts.N
    subgraph = reset_subg(mcts.state)
    rw_path = []

    while mcts.C != None: 
        if np.random.rand() < args.restart: 
            mcts = reset_agent(mcts)
            rw_path.append(-1)
        else: 
            children = mcts.C
            if len(children) == 0: 
                mcts = reset_agent(mcts)
                rw_path.append(-1)
                if mcts.parent == None:
                    break
            else:
                try:
                    if (rw_path[-1] == -1) and (len(mcts.C)>=2):
                        s = np.argmax([UCB1(children[i].Vi, N, children[i].N, args.c1) for i in children])
                        nlst = list(range(0, len(mcts.C)))
                        nlst.remove(s)
                        s = np.random.choice(nlst, 1)[0]    
                    else:
                        s = np.argmax([UCB1(children[i].Vi, N, children[i].N, args.c1) for i in children])
                        
                except IndexError:
                    s = np.argmax([UCB1(children[i].Vi, N, children[i].N, args.c1) for i in children])
                
                subgraph.add_edge(mcts.state, mcts.C[s].state)
                mcts = mcts.C[s]
                rw_path.append(s)

    return mcts, subgraph, rw_path

def simulate(args, subgraph, initial_nd, model, G, bf_top5_idx, bf_dist, x, edge_idx):
    

    edges_to_perturb = list(subgraph.edges())
    value = importance(args, model, x, edge_idx, bf_top5_idx, bf_dist, edges_to_perturb, initial_nd)
   
    return value

def backprop(mcts, rw_path, value):
    
    mcts = reset_agent(mcts)
    if value > mcts.V:
        mcts.V = value
        mcts.Vi = mcts.V
    mcts.N += 1

    
    for i in rw_path:
        if i == -1:
            mcts = reset_agent(mcts)
        else:
            mcts = mcts.C[i]
            if value > mcts.V:
                mcts.V = value
                mcts.Vi = mcts.V
            mcts.N += 1

def reset_agent(mcts):
    while mcts.parent != None:
        mcts = mcts.parent
    return mcts

def reset_subg(initial_nd):
    subgraph = nx.Graph()
    subgraph.add_node(initial_nd)
    return subgraph

def khop_sampling(G,initial_nd):
    subgraph = nx.ego_graph(G, initial_nd, radius=2)
    return subgraph

