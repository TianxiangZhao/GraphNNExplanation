import torch
from torch_geometric.data import Data
from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.utils import from_networkx
import random
import numpy as np
import networkx as nx
import ipdb
import pickle as pkl

def house():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                               [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]])
    label = torch.tensor([1, 1, 2, 2, 3])
    return edge_index, label

def cycle():
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1],
                               [1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1, 0]])
    label = torch.tensor([1, 1, 1, 1, 1, 1])
    return edge_index, label


def grid():
    G = nx.grid_2d_graph(3,3)
    edge_index = from_networkx(G).edge_index
    label = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
    return edge_index, label

def tree(r=2, height=3):#15 nodes
    g = nx.balanced_tree(r, height)
    data = from_networkx(g)
    edge_index = data.edge_index
    label = torch.zeros(data.num_nodes)

    return edge_index, label

def ladder(length=7): #14 nodes
    g = nx.ladder_graph(n=length)
    data = from_networkx(g)
    edge_index = data.edge_index
    label = torch.zeros(data.num_nodes)

    return edge_index, label

def wheel(n=15): #15 nodes
    g= nx.wheel_graph(n=n)
    data = from_networkx(g)
    edge_index = data.edge_index
    label = torch.zeros(data.num_nodes)

    return edge_index, label





def syn_BA_shapes(connection_distribution: str = "uniform",):
    
    assert connection_distribution in ['random', 'uniform']
    
    # Build the Barabasi-Albert graph:
    num_nodes = 300
    edge_index = barabasi_albert_graph(num_nodes, num_edges=5)
    edge_label = torch.zeros(edge_index.size(1), dtype=torch.int64)
    node_label = torch.zeros(num_nodes, dtype=torch.int64)
    
    # Select nodes to connect shapes:
    num_houses = 80
    if connection_distribution == 'random':
        connecting_nodes = torch.randperm(num_nodes)[:num_houses]
    else:
        step = num_nodes // num_houses
        connecting_nodes = torch.arange(0, num_nodes, step)

    # Connect houses to Barabasi-Albert graph:
    edge_indices = [edge_index]
    edge_labels = [edge_label]
    node_labels = [node_label]
    for i in range(num_houses):
        house_edge_index, house_label = house()

        edge_indices.append(house_edge_index + num_nodes)
        edge_indices.append(torch.tensor([[int(connecting_nodes[i]), num_nodes],
                          [num_nodes, int(connecting_nodes[i])]]))

        edge_labels.append(
            torch.ones(house_edge_index.size(1), dtype=torch.long))
        edge_labels.append(torch.zeros(2, dtype=torch.long))

        node_labels.append(house_label)

        num_nodes += 5

    edge_index = torch.cat(edge_indices, dim=1)
    edge_label = torch.cat(edge_labels, dim=0)
    node_label = torch.cat(node_labels, dim=0)

    
    print('setting expl mask as the first node labeled as 1 in each house structure')
    x = torch.ones((num_nodes, 10), dtype=torch.float)
    expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
    expl_mask[torch.arange(400, num_nodes, 5)] = True

    data = Data(x=x, edge_index=edge_index, y=node_label,
                expl_mask=expl_mask, edge_label=edge_label)

            
    return data

def syn_Tree_cycle(height=8, r=2, connection_distribution: str = "random",):
    '''
    generate Tree_cycle graph for explanation evaluation

    Args:
        height: height of the base tree
        r: branches of each node
    '''
    assert connection_distribution in ['random', 'uniform']
    
    # Build the base tree graph:
    g = nx.balanced_tree(r, height)
    base_nodes = len(g.nodes())
    data = from_networkx(g)
    edge_index = data.edge_index
    edge_label = torch.zeros(edge_index.size(1), dtype=torch.int64)
    node_label = torch.zeros(base_nodes, dtype=torch.int64)

    # Select nodes to connect shapes:
    num_cycles = 80
    if connection_distribution == 'random':
        connecting_nodes = torch.randperm(base_nodes)[:num_cycles]
    else:
        step = base_nodes // num_cycles
        connecting_nodes = torch.arange(0, base_nodes, step)

    # Connect houses to base tree:
    edge_indices = [edge_index]
    edge_labels = [edge_label]
    node_labels = [node_label]
    num_nodes = base_nodes
    for i in range(num_cycles):
        cycle_edge_index, cycle_label = cycle()

        edge_indices.append(cycle_edge_index + num_nodes)
        edge_indices.append(torch.tensor([[int(connecting_nodes[i]), num_nodes],
                          [num_nodes, int(connecting_nodes[i])]]))

        edge_labels.append(
            torch.ones(cycle_edge_index.size(1), dtype=torch.long))
        edge_labels.append(torch.zeros(2, dtype=torch.long))

        node_labels.append(cycle_label)

        num_nodes += 6

    edge_index = torch.cat(edge_indices, dim=1)
    edge_label = torch.cat(edge_labels, dim=0)
    node_label = torch.cat(node_labels, dim=0)

    
    print('setting expl mask as the attached nodes')
    x = torch.ones((num_nodes, 10), dtype=torch.float)
    expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
    expl_mask[torch.arange(base_nodes, num_nodes, 6)] = True

    data = Data(x=x, edge_index=edge_index, y=node_label,
                expl_mask=expl_mask, edge_label=edge_label)
            
    return data


def syn_Tree_grid(height=8, r=2, connection_distribution: str = "random",):
    '''
    generate Tree_cycle graph for explanation evaluation

    Args:
        height: height of the base tree
        r: branches of each node
    '''
    assert connection_distribution in ['random', 'uniform']
    
    # Build the base tree graph:
    g = nx.balanced_tree(r, height)
    base_nodes = len(g.nodes())
    data = from_networkx(g)
    edge_index = data.edge_index
    edge_label = torch.zeros(edge_index.size(1), dtype=torch.int64)
    node_label = torch.zeros(base_nodes, dtype=torch.int64)

    # Select nodes to connect shapes:
    num_grid = 80
    if connection_distribution == 'random':
        connecting_nodes = torch.randperm(base_nodes)[:num_grid]
    else:
        step = base_nodes // num_grid
        connecting_nodes = torch.arange(0, base_nodes, step)

    # Connect houses to base tree:
    edge_indices = [edge_index]
    edge_labels = [edge_label]
    node_labels = [node_label]
    num_nodes = base_nodes
    for i in range(num_grid):
        grid_edge_index, grid_label = grid()

        edge_indices.append(grid_edge_index + num_nodes)
        edge_indices.append(torch.tensor([[int(connecting_nodes[i]), num_nodes],
                          [num_nodes, int(connecting_nodes[i])]]))

        edge_labels.append(
            torch.ones(grid_edge_index.size(1), dtype=torch.long))
        edge_labels.append(torch.zeros(2, dtype=torch.long))

        node_labels.append(grid_label)

        num_nodes += 9

    edge_index = torch.cat(edge_indices, dim=1)
    edge_label = torch.cat(edge_labels, dim=0)
    node_label = torch.cat(node_labels, dim=0)

    print('setting expl mask as the attached nodes')
    x = torch.ones((num_nodes, 10), dtype=torch.float)
    expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
    expl_mask[torch.arange(base_nodes, num_nodes, 9)] = True

    data = Data(x=x, edge_index=edge_index, y=node_label,
                expl_mask=expl_mask, edge_label=edge_label)
            
    return data

def syn_dataset(name):
    '''
    generate synthetic dataset in the form of torch_geometric.data.Data
    '''
    if name == 'BA_shapes':
        data = syn_BA_shapes()
    if name == 'Tree_cycle':
        data = syn_Tree_cycle()
    if name == 'Tree_grid':
        data = syn_Tree_grid()
    

    return data

def syn_spMotif(mix_ratio, tot_number = 3000):
    #generate spurious motif dataset, for graph classification
    graph_gens = [tree, ladder, wheel]
    motif_gens = [cycle, house, grid]

    graphs = []
    accord_num = int(tot_number/3*(mix_ratio+(1-mix_ratio)/3))
    naccord_num = int(tot_number/3*(1-mix_ratio)*(1/3))
    for base_id in range(3):
        for motif_id in range(3):
            number = accord_num if base_id==motif_id else naccord_num
            
            for i in range(number):
                edge_indices=[]
                edge_labels=[]
                base_edge_index, base_label = graph_gens[base_id]()
                motif_edge_index, motif_label = motif_gens[motif_id]()

                base_nodes = base_label.shape[0]
                num_nodes = base_nodes+motif_label.shape[0]
                edge_indices.append(base_edge_index)
                edge_indices.append(motif_edge_index + base_nodes)
                connect_src = random.randint(0,base_nodes-1)
                connect_tgt = random.randint(base_nodes, num_nodes-1)
                #connect_tgt = base_nodes
                edge_indices.append(torch.tensor([[connect_src, connect_tgt],
                          [connect_tgt, connect_src]]))

                edge_labels.append(torch.zeros(base_edge_index.size(1), dtype=torch.long))
                edge_labels.append(torch.ones(motif_edge_index.size(1), dtype=torch.long))
                edge_labels.append(torch.zeros(2, dtype=torch.long))
            
                edge_index = torch.cat(edge_indices, dim=1)
                edge_label = torch.cat(edge_labels, dim=0)

                x = torch.ones((num_nodes, 10), dtype=torch.float)
                #x[:base_nodes,:] = 0.0
                expl_mask = True
                y = motif_id

                data = Data(x=x, edge_index=edge_index, y=y,
                            expl_mask=expl_mask, edge_label=edge_label)
                graphs.append(data)

    return graphs



def load_dataset(name):
    '''
    generate synthetic dataset in the form of torch_geometric.data.Data
    '''
    if name == 'BA_shapes':
        data_path = './datasets/temp/syn1.pkl'
    if name == 'Tree_cycle':
        data_path = './datasets/temp/syn3.pkl'
    if name == 'Tree_grid':
        data_path = './datasets/temp/syn4.pkl'
    
    with open(data_path, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)
    labels = y_train
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]
    labels = labels.argmax(axis=-1)
    edge_index = np.stack(adj.nonzero())

    
    data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(edge_index), y=torch.tensor(labels, dtype=torch.int64),
        train_mask = torch.tensor(train_mask,dtype=torch.bool),val_mask = torch.tensor(val_mask,dtype=torch.bool),test_mask = torch.tensor(test_mask,dtype=torch.bool))


    return data


def split_graph(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    '''
    create node masks on the graph
    Args:
        data (torch_geometric.data.Data), input graph
    Returns:
        train_mask (torch.tensor) of shape (node_number,), contains boolean masks over nodes
    '''
    num_classes = len(set(data.y.tolist()))
    
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []

    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (data.y==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        c_num_mat[i,0] = int(c_num*train_ratio)
        c_num_mat[i,1] = int(c_num*val_ratio)
        c_num_mat[i,2] = int(c_num*test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    train_mask = data.y.new(data.y.shape).fill_(0).bool()
    train_mask[train_idx] = 1
    val_mask = data.y.new(data.y.shape).fill_(0).bool()
    val_mask[val_idx] = 1
    test_mask = data.y.new(data.y.shape).fill_(0).bool()
    test_mask[test_idx] = 1


    return train_mask, val_mask, test_mask