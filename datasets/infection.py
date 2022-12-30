import torch
import pickle as pkl
import os
from torch_geometric.data import Data,InMemoryDataset
from torch_geometric.utils import from_networkx
from typing import Optional, Callable
import networkx as nx
import random
try:
    from . import data_utils     # "myapp" case
except:
    import data_utils            # "__main__" case
import ipdb
import numpy as np

class infection(InMemoryDataset):
    """ Adapted from https://github.com/m30m/gnn-explainability, KDD2021 paper

    Args:
        num_graphs:
        max_dist: largest available distance. greater distance would be set as max_dist+1. In total, has class number: max_dist+2

    """
    def __init__(self, num_graphs=1, name='infection', max_dist=3,
                 transform: Optional[Callable] = None, pre_transform=None):
        self.num_graphs = num_graphs
        self.max_dist = max_dist
        self.node_num = 1000
        self.edge_prob = 0.004
        
        super().__init__('./datasets/{}'.format(name), transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['generated_graph0.pt']

    @property
    def processed_file_names(self):
        return ['graph.pt']

    def create_dataset(self):
        max_dist = self.max_dist  # anything larger than max_dist has a far away label
        g = nx.erdos_renyi_graph(self.node_num, self.edge_prob, directed=True)
        N = len(g.nodes())
        infected_nodes = random.sample(g.nodes(), 50)

        #create graph with distance to infected nodes
        g.add_node('X')  # dummy node for easier computation, will be removed in the end
        for u in infected_nodes:
            g.add_edge('X', u)
        shortest_path_length = nx.single_source_shortest_path_length(g, 'X')
        unique_solution_nodes = []
        unique_solution_explanations = []
        labels = []
        features = np.zeros((N, 2))
        for i in range(N):
            if i == 'X':
                continue
            length = shortest_path_length.get(i, 100) - 1  # 100 is inf distance
            labels.append(min(max_dist + 1, length))
            col = 0 if i in infected_nodes else 1
            features[i, col] = 1
            if 0 < length <= max_dist:
                path_iterator = iter(nx.all_shortest_paths(g, 'X', i))
                unique_shortest_path = next(path_iterator)
                if next(path_iterator, 0) != 0:
                    continue
                unique_shortest_path.pop(0)  # pop 'X' node
                if len(unique_shortest_path) == 0:
                    continue
                unique_solution_explanations.append(unique_shortest_path)
                unique_solution_nodes.append(i)
        g.remove_node('X')

        data = from_networkx(g)
        data.x = torch.tensor(features, dtype=torch.float)
        data.y = torch.tensor(labels)
        data.expl_mask = torch.zeros(N, dtype=torch.bool)
        data.expl_mask[torch.tensor(unique_solution_nodes)] = True

        data.edge_expl_path = data.x.new(N,self.max_dist+2).fill_(-1) #path from current node to nearest infected node
        for expl_node, expl_path in zip(unique_solution_nodes, unique_solution_explanations):
            path_len = len(expl_path)
            for j,tgt_node in enumerate(expl_path):
                data.edge_expl_path[expl_node,path_len-1-j] = tgt_node

        data.num_classes = 1 + max_dist + 1

        return data

    def download(self):
        # for synthetic dataset, directly generate it
        for raw_path in self.raw_paths:
            data = self.create_dataset()
            train_mask, val_mask, test_mask = data_utils.split_graph(data)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            torch.save(data,raw_path)

    def process(self):
        #load data
        data_list = [torch.load(f) for f in self.raw_paths]
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    data = infection()

    ipdb.set_trace()

    print('data created')
