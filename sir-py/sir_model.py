import random
import numpy as np
import networkx as nx
import pickle as pkl

def sir_model(g):
    for i in g :
        if g.node[i]['state'] == 1:
            nei = g.neighbors(i)
            for j in nei:
                p = random.random()
                if p > 0.7:
                    g.node[j]['state'] = 1
            if g.node[i]['tns'] > tn:
                g.node[i]['state'] = 2
                g.node[i]['tns'] = 0
                g.node[i]['tms'] += 1
            else:
                g.node[i]['tns'] += 1

        if g.node[i]['state'] == 2:
            if g.node[i]['tms'] > tm:
                g.node[i]['state'] = 0
                g.node[i]['tms'] = 0
            else:
                g.node[i]['tms'] += 1
    return g

def node_color(g):
    for v in g:
        if g.node[v]['state'] == 0:
            g.node[v]['color'] = 'white'
        if g.node[v]['state'] == 1:
            g.node[v]['color'] = 'red'
        if g.node[v]['state'] == 2:
            g.node[v]['color'] = 'green'
    node_color = [g.node[v]['color'] for v in g]
    return node_color

#generate the net
d = 100
g = nx.random_graphs.watts_strogatz_graph(d, 8, 0.1)
A = nx.adjacency_matrix(g).astype(np.float32)

#train data
for i in g.node.keys():
    g.node[i]['state'] = 0
    g.node[i]['tns'] = 0
    g.node[i]['tms'] = 0

nodes = list(g.node.keys())
init_n = random.choice(nodes)
g.node[init_n]['state'] = 1

tn = 2
tm = 10

n = 100 #process step
x_data = []
y_data = []
for step in range(n):
    x_data.append([g.node[v]['state'] for v in g])
    g = sir_model(g)
    y_data.append([g.node[v]['state'] for v in g])

x_data = np.array(x_data).astype(np.float32)
y_data = np.array(y_data)

with open('./data/x_data.pkl','wb') as f1:
    pkl.dump(x_data, f1)

with open('./data/y_data.pkl','wb') as f2:
    pkl.dump(y_data, f2)

with open('./data/adj_data.pkl','wb') as f3:
    pkl.dump(A, f3)

