from __future__ import division
from __future__ import print_function

import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util.models import GCN
from util.utils import accuracy, sparse_mx_to_torch_sparse_tensor

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
f1 = open('adj.pkl', 'rb')
adj = pkl.load(f1)
adj = sparse_mx_to_torch_sparse_tensor(adj)
f2 = open('features.pkl', 'rb')
features = pkl.load(f2)
features = torch.FloatTensor(features)
f3 = open('labels.pkl', 'rb')
labels = pkl.load(f3)
labels = torch.FloatTensor(labels)

idx_train = range(700)
idx_val = range(700, 800)
idx_test = range(800, 1000)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

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
    output = model(features, adj)
    loss_train = F.mse_loss(output[idx_train], labels[idx_train])
    #acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    loss_val = F.mse_loss(output[idx_val], labels[idx_val])
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
    loss_test = F.mse_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


# Train model
t_total = time.time()
l1_train = []
for epoch in range(epochs):
    l1_train.append(train(epoch))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

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

# Testing
test()