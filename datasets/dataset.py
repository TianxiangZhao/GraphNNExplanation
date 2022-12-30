import torch
import pickle as pkl
import os
from torch_geometric.data import Data,InMemoryDataset
from torch_geometric.utils import barabasi_albert_graph
from typing import Optional, Callable

import datasets.data_utils as data_utils
import ipdb

class SyncShapes(InMemoryDataset):
    """ Adapted from torch_geometric dataset: BAShapes
    The BA-Shapes dataset from the `"GNNExplainer: Generating Explanations
    for Graph Neural Networks" <https://arxiv.org/pdf/1903.03894.pdf>`_ paper,
    containing a Barabasi-Albert (BA) graph with 300 nodes and a set of 80
    "house"-structured graphs connected to it.

    Args:
        connection_distribution (string, optional): Specifies how the houses
            and the BA graph get connected. Valid inputs are :obj:`"random"`
            (random BA graph nodes are selected for connection to the houses),
            and :obj:`"uniform"` (uniformly distributed BA graph nodes are
            selected for connection to the houses). (default: :obj:`"random"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(self, name, 
                 transform: Optional[Callable] = None, pre_transform=None):
                 
        self.name = name
        super().__init__('./datasets/{}'.format(name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['generated_graph.pt']

    @property
    def processed_file_names(self):
        return ['graph.pt']

    def download(self):
        # for synthetic dataset, directly generate it
        data = data_utils.syn_dataset(self.name)
        train_mask, val_mask, test_mask = data_utils.split_graph(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        torch.save(data,self.raw_paths[0])

    def process(self):
        #load data
        data_list = [torch.load(f) for f in self.raw_paths]
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
