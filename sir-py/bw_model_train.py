
import sys
sys.path.append('..')
import pickle as pkl
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util.models import GCN
from util.utils import sparse_mx_to_torch_sparse_tensor, accuracy
from util.utils import save_file, read_file

#setting
hidden = 16
dropout = 0.5
lr = 2e-3
weight_decay = 5e-4
epochs = 20
seed = 42
no_cuda = False
fastmode = False

cuda = not no_cuda and torch.cuda.is_available()

np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Model and optimizer
model = GCN(nfeat=1,
            nhid=hidden,
            nclass=2,
            dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

def train(iteration):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output, labels)
    acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        model.eval()
        output = model(features, adj)

    los_val_ = []
    if (iteration+1) % 10 == 0:
        loss_val = F.nll_loss(output, labels)
        acc_val = accuracy(output, labels)
        los_val_.append(loss_val.data[0])
        print('Epoch:{:04d} Val loss:{:.4f} Val acc:{:.4f}'.format(epoch+1, loss_val.data[0], acc_val.data[0]))

    return loss_train.data[0],acc_train.data[0],los_val_


def test(test_feature,test_label):
    model.eval()
    output = model(test_feature, adj)
    print(output.max(1)[1].data)
    print(test_label.data)
    acc_test = accuracy(output, test_label)
    loss_test = F.nll_loss(output, test_label)
    print("Test set results: loss={:.4f} test acc={:.4f}".format(loss_test.data[0],acc_test.data[0]))

#load data
x_data = read_file('./data/bw_x_data','pkl')
y_data = read_file('./data/bw_y_data','pkl')
A = read_file('./data/bw_adj_data','pkl')

time_step = len(x_data)
nodes = len(x_data[0])

adj = sparse_mx_to_torch_sparse_tensor(A)
adj = Variable(adj)

# Train model
t_total = time.time()
train_step = time_step - 0
ep_avg_loss = []
ep_avg_acc = []
ep = []
los_val = []
for epoch in range(epochs):
    lo_train = []
    lo_acc = []
    for iteration in range(train_step):
        features = np.array(x_data[iteration]).reshape([nodes, 1])
        labels = np.array(y_data[iteration])
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        features, labels = Variable(features), Variable(labels)
        if cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
        train_loss, train_acc, loss_val = train(iteration)
        lo_train.append(train_loss)
        lo_acc.append(train_acc)
    avg_loss = np.mean(lo_train)
    avg_acc = np.mean(lo_acc)
    ep_avg_loss.append(avg_loss)
    ep_avg_acc.append(avg_acc)
    ep.append(epoch)
    los_val.append(np.mean(loss_val))
    print('Epoch:{:04d} Avg Loss:{:.4f} Avg Acc:{:.4f}'.format(epoch, avg_loss, avg_acc))

save_file('epoch',ep,'txt')
save_file('avg_loss',ep_avg_loss,'txt')
save_file('avg_acc',ep_avg_acc,'txt')
save_file('val_loss', los_val, 'txt')

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test_feature = np.array(x_data[6]).reshape([nodes, 1])
test_label = np.array(y_data[6])
test_feature = torch.FloatTensor(test_feature)
test_label = torch.LongTensor(test_label)
test_feature, test_label = Variable(test_feature), Variable(test_label)
test(test_feature, test_label)
