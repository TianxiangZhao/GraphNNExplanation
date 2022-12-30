import torch
import pickle as pkl
import os
from torch_geometric.data import Data,InMemoryDataset
from typing import Optional, Callable
import ipdb
try:
    from . import data_utils     # "myapp" case
except:
    import data_utils            # "__main__" case

class SyncGraphs(InMemoryDataset):
    """ Generate synthetic graph-cls task for explanation

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
    def __init__(self, name, mix_ratio=0.5, tot_num=3000,
                 transform: Optional[Callable] = None, pre_transform=None):
                 
        self.name = name
        self.mix_ratio = mix_ratio
        self.tot_num = tot_num
        super().__init__('./datasets/{}_{}'.format(name, mix_ratio), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['generated_graph.pt']

    @property
    def processed_file_names(self):
        return ['graph.pt']

    def download(self):
        # for synthetic dataset, directly generate it
        if self.name == 'SpuriousMotif':
            graphs = data_utils.syn_spMotif(self.mix_ratio,tot_number=self.tot_num)
            torch.save(graphs,self.raw_paths[0])

    def process(self):
        #load data
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    ipdb.set_trace()
    data = SyncGraphs('SpuriousMotif',0.3, tot_num=10)
    print('data synthesized')
