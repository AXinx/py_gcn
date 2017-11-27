from __future__ import division
from __future__ import print_function

import math
import pickle as pkl
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from util.utils import sparse_mx_to_torch_sparse_tensor

# Load data
d = 1000
g = nx.random_graphs.watts_strogatz_graph(d, 8, 0.1)
adj = nx.adjacency_matrix(g).astype(np.float32)
with open('adj.pkl','wb') as f:
    pkl.dump(adj, f)
adj = sparse_mx_to_torch_sparse_tensor(adj)
features = np.array([random.random() for _ in range(d)]).reshape([d,1])
with open('features.pkl', 'wb') as f:
    pkl.dump(features, f)
features = torch.FloatTensor(features)

def caculate_gcn(x, adj):
    nfeat = features.shape[1]
    nhid = 16
    nclass = 1
    dropout = 0.5
    x = F.relu(GraphConvolution(x, adj, nfeat, nhid))
    x = F.dropout(x, dropout)
    x = GraphConvolution(x.data, adj, nhid, nclass)
    return x

def GraphConvolution(input, adj, in_features, out_features):
    weight = torch.rand(in_features, out_features)
    #with open('weight_'+str(out_features)+'.pkl', 'wb') as f:
    #    pkl.dump(weight, f)
    bias = torch.rand(out_features)
    stdv = 1. / math.sqrt(weight.size(1))
    weight = weight.uniform_(-stdv, stdv)
    bias = bias.uniform_(-stdv, stdv)
    print(weight)
    print(bias)
    #with open('bias_'+str(out_features)+'.pkl','wb') as f:
    #    pkl.dump(bias, f)
    support = torch.mm(input, weight)
    output = SparseMM(adj, support)
    if bias is not None:
        return output + bias
    else:
        return output

def SparseMM(matrix1, matrix2):
    return torch.mm(matrix1, matrix2)

output = caculate_gcn(features, adj)
print(output)
with open('labels.pkl', 'wb') as f:
    pkl.dump(output, f)