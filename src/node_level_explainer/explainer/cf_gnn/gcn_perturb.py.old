import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ...utils.utils import get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from ...oracles.models.gcn import GraphConvolution, GCNSynthetic

class GraphConvolutionPerturb(nn.Module):
	"""
	Similar to GraphConvolution except includes P_hat
	It is referred in the original paper as g(A_v, X_v, W; P) = $softmax [\sqrt(D^{line}_v]) (P*A_v + I) \sqrt(D^{line}_v]) X_v * W$
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolutionPerturb, self).__init__()

		self.in_features = in_features
		self.out_features = out_features

		# Build a random weight tensor
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))

		# If bias then build a random tensor for bias
		if bias is not None:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)

	def forward(self, input: torch.Tensor, adj: torch.Tensor)->torch.Tensor:
		"""
		Forward pass

		args:
			input: the features of the graph

			adj: the adjacency matrix
		"""

		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		return output if self.bias is None else output + self.bias

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
		       + str(self.in_features) + ' -> ' \
		       + str(self.out_features) + ')'

class GCNSyntheticPerturb(nn.Module):
	"""
	3-layer GCN used in GNN Explainer synthetic tasks
	"""
	def __init__(self, cfg, nfeat: int, nhid: int, nout: int, nclass: int, adj, dropout, beta, edge_additions=False):
		super(GCNSyntheticPerturb, self).__init__()

		self.adj = adj
		self.nclass = nclass
		self.beta = beta
		self.num_nodes = self.adj.shape[0]
		self.edge_additions = edge_additions 
		# P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
		self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

		# Perturbation matrix initialization
		if self.edge_additions:
			self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
		else:
			self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

		self.cfg = cfg
		self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
		self.gc2 = GraphConvolutionPerturb(nhid, nhid)
		self.gc3 = GraphConvolution(nhid, nout)
		self.lin = nn.Linear(nhid + nhid  + nout, nclass)
		self.dropout = dropout
  
		self.I = torch.eye(self.num_nodes, device=self.cfg.device)

	def forward(self, x, sub_adj):

		self.sub_adj = sub_adj
		# Same as normalize_adj in utils.py except includes P_hat in A_tilde
		self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes, device=self.cfg.device)      # Ensure symmetry

		# Initizlize the adjacency matrix with self-loops
		A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
		A_tilde.requires_grad = True
  
		A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + self.I      # Use sigmoid to bound P_hat in [0,1]

		# D_tilde is the degree matrix
		D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
		# Compute the degree matrix and its inverse square root
		D_tilde = torch.pow(D_tilde, -0.5)
		D_tilde[torch.isinf(D_tilde)] = 0  # Set inf values to 0

		# Compute the normalized adjacency matrix
		norm_adj = D_tilde @ A_tilde @ D_tilde

		x1 = F.relu(self.gc1(x, norm_adj))
		x1 = F.dropout(x1, self.dropout, training=self.training)
		x2 = F.relu(self.gc2(x1, norm_adj))
		x2 = F.dropout(x2, self.dropout, training=self.training)
		x3 = self.gc3(x2, norm_adj)
		x = self.lin(torch.cat((x1, x2, x3), dim=1))
		return F.log_softmax(x, dim=1)


	def forward_prediction(self, x):

		# Same as forward but uses P instead of P_hat ==> non-differentiable
		# but needed for actual predictions
		self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat

		A_tilde = self.P * self.adj + self.I

		D_tilde = get_degree_matrix(A_tilde)
		# Raise to power -1/2, set all infs to 0s
		# Compute the degree matrix and its inverse square root
		D_tilde = torch.pow(D_tilde, -0.5)
		D_tilde[torch.isinf(D_tilde)] = 0  # Set inf values to 0

		# Compute the normalized adjacency matrix
		norm_adj = D_tilde @ A_tilde @ D_tilde

		x1 = F.relu(self.gc1(x, norm_adj))
		x1 = F.dropout(x1, self.dropout, training=self.training)
		x2 = F.relu(self.gc2(x1, norm_adj))
		x2 = F.dropout(x2, self.dropout, training=self.training)
		x4 = F.relu(self.gc2(x2, norm_adj))
		x4 = F.dropout(x4, self.dropout, training=self.training)
		x3 = self.gc3(x4, norm_adj)
		x = self.lin(torch.cat((x1, x2, x3), dim=1))
		return F.log_softmax(x, dim=1), self.P

	def loss(self, graph, output, y_node_non_differentiable):

		node_to_explain = graph.new_idx

		y_node_predicted = output[node_to_explain].unsqueeze(0)
		y_node_oracle_original = graph.y[node_to_explain]
		y_target = graph.targets[node_to_explain].unsqueeze(0)

		pred_same = (y_node_non_differentiable == y_node_oracle_original).float()

		cf_adj = self.P * self.adj
		cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

		# Want negative in front to maximize loss instead of minimizing it to find CFs
		#loss_pred =  F.nll_loss(output, 1-y_pred_orig)
		loss_pred =  F.cross_entropy(y_node_predicted, y_target)
		loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2     # Number of edges changed (symmetrical)
	
		# Zero-out loss_pred with pred_same if prediction flips
		loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

		results = {
			"loss_total":  loss_total.item(),
			"loss_pred": loss_pred.item(),
			"loss_graph_dist": loss_graph_dist.item(),
		}

		return loss_total, results, cf_adj
