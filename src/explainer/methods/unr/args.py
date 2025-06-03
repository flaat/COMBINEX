import argparse

class Config:
    def __init__(self, path='./result/', dataset='Cora', model='gcn', perturb=0.0, maxiter=1000, explainer='mctsrestart', task='node', gpu='0', hidden_dim=128, c1=1.0, neighbors_cnt=5, restart=0.2, iter=500, lr=0.001):
        self.path = path
        self.dataset = dataset
        self.model = model
        self.perturb = perturb
        self.maxiter = maxiter
        self.explainer = explainer
        self.task = task
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.c1 = c1
        self.neighbors_cnt = neighbors_cnt
        self.restart = restart
        self.iter = iter
        self.lr = lr

def parse_args():
    config = Config()
    return config
