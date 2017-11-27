import networkx as nx
import numpy as np
import pickle as pkl
import random
import matplotlib.pyplot as plt

# setting
beta = 0.2
gemma = 0.2
tr = 10

# 规则
# I状态节点以概率beta感染邻居节点
# I状态节点到达时间tr之后以gamme概率死亡，1-gamme概率治愈

def sir_model(g):
    for i in g:
        if g.node[i]['state'] == 1:
            nei = g.neighbors(i)
            for each in nei:
                if g.node[each]['state'] == 0:
                    if random.random() < beta:
                        g.node[each]['state'] = 1
                        g.node[each]['time'] = tr
                        g.node[each]['color'] = 'red'
                if g.node[each]['state'] == 1:
                    g.node[each]['time'] -= 1
        if g.node[i]['time'] == 0:
            g.node[i]['state'] = 2
            if random.random() < gemma:
                g.node[i]['color'] = 'black'
            else:
                g.node[i]['color'] = 'yellow'
    return g

# generate the net
d = 200
g = nx.random_graphs.watts_strogatz_graph(d, 4, 0.1)
A = nx.adjacency_matrix(g).astype(np.float32)

# train data
for i in g.node.keys():
    g.node[i]['state'] = 0
    g.node[i]['time'] = 100
    g.node[i]['color'] = 'green'

nodes = list(g.node.keys())
init_n = random.choice(nodes)
g.node[init_n]['state'] = 1
g.node[init_n]['time'] = tr
g.node[init_n]['color'] = 'red'

nodes = nx.number_of_nodes(g)
edges = nx.number_of_edges(g)
average_dgree = 2 * edges / nodes

print(nodes)
print(edges)
print(average_dgree)

time_step = 30 #大概30步就稳定了
x_data = []
y_data = []
sus_r = []
inf_r = []
r_r = []
step = []
for i in range(time_step):
    x_data.append([g.node[v]['state'] for v in g])
    g = sir_model(g)
    y_data.append([g.node[v]['state'] for v in g])

    state = [g.node[v]['state'] for v in g]
    sus_node = state.count(0)
    inf_node = state.count(1)
    r_node = state.count(2)

    sus_rate = sus_node / d
    inf_rate = inf_node / d
    r_rate = r_node / d

    step.append(i)
    sus_r.append(sus_rate)
    inf_r.append(inf_rate)
    r_r.append(r_rate)

    print('-------')
    print(i)
    print(sus_rate)
    print(inf_rate)
    print(r_rate)

plt.figure()
plt.title('SW: n=200, <k> = 4')
plt.plot(step, sus_r, label='S')
plt.plot(step, inf_r, label='I')
plt.plot(step, r_r, label='R')
plt.legend(loc='best')
plt.show()
plt.savefig('SW-SIR.png')


x_data = np.array(x_data).astype(np.float64)
y_data = np.array(y_data)

with open('./data/x_data.pkl','wb') as f1:
    pkl.dump(x_data, f1)

with open('./data/y_data.pkl','wb') as f2:
    pkl.dump(y_data, f2)

with open('./data/adj_data.pkl','wb') as f3:
    pkl.dump(A, f3)
