from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util.models import GCN
from util.utils import sparse_mx_to_torch_sparse_tensor

#setting
hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 100
seed = 42
no_cuda = False
fastmode = False

cuda = not no_cuda and torch.cuda.is_available()

np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load data
d = 1000
g = nx.random_graphs.watts_strogatz_graph(d, 8, 0.1)
adj = nx.adjacency_matrix(g).astype(np.float32)
adj = sparse_mx_to_torch_sparse_tensor(adj)
#features = np.array(np.ones(d)).reshape([d,1])
features = np.array([random.random() for _ in range(d)]).reshape([d,1])
features = torch.FloatTensor(features)
#labels = np.array(np.ones(d)+1).reshape([d,1])
labels = np.array([random.random() for _ in range(d)]).reshape([d,1])
labels = torch.FloatTensor(labels)

print(adj)

#adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=1,
            dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    print(epoch)
    output = model(features, adj)
    loss_train = F.mse_loss(output, labels)
    #acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.mse_loss(output, labels)
    #acc_val = accuracy(output, labels)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          #'acc_train: {:.4f}'.format(acc_train.data[0]),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          #'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.mse_loss(output, labels)
    #acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]))
          #"accuracy= {:.4f}".format(acc_test.data[0]))


# Train model
t_total = time.time()
l1_train = []
for epoch in range(epochs):
    l1_train.append(train(epoch))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

'''
l_train = []
for each in l1_train:
    each = each.data.numpy()
    l_train.append(each)
print(l_train)
plt.figure()
s = len(l_train)
x = list(range(1,s+1))
print(x)
plt.plot(x,l_train)
plt.show()
plt.savefig('loss.jpg')
'''
# Testing
#test()