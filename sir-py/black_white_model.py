
import networkx as nx
import numpy as np
import random
import pickle as pkl

d = 1000
g = nx.random_graphs.watts_strogatz_graph(d, 4, 0.1)
A = nx.adjacency_matrix(g).astype(np.float32)

x_data = []
y_data = []

for i in g.node:
    g.node[i]['state'] = 0

nodes = list(g.node.keys())
init_one = random.choice(nodes)
g.node[init_one]['state'] = 1

def processing(g):
    black_node = []
    for i in g.node:
        if g.node[i]['state'] == 1:
            nei = g.neighbors(i)
            black_node += nei
    for each in black_node:
        if g.node[each]['state'] != 1:
            g.node[each]['state'] = 1

process_step = 10

for i in range(process_step):
    node_state_start = [g.node[v]['state'] for v in g]
    x_data.append(node_state_start)
    processing(g)
    node_state_end = [g.node[v]['state'] for v in g]
    y_data.append(node_state_end)


x_data = np.array(x_data).astype(np.float32)
y_data = np.array(y_data)

with open('./data/bw_x_data.pkl','wb') as f1:
    pkl.dump(x_data, f1)

with open('./data/bw_y_data.pkl','wb') as f2:
    pkl.dump(y_data, f2)

with open('./data/bw_adj_data.pkl','wb') as f3:
    pkl.dump(A, f3)

