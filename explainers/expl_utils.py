import torch
import ipdb

def to_sym(edge_index, edge_weights, node_num):
    #change edge weights to symmetric
    edge_weights = edge_weights.squeeze()
    weight_dense = (torch.sparse_coo_tensor(edge_index, edge_weights, (node_num,node_num))).to_dense()
    weight_dense = (weight_dense + weight_dense.transpose(-1,-2))/2

    #get symmetric edge_index
    edge_dense = (torch.sparse_coo_tensor(edge_index, torch.ones(edge_weights.shape).to(edge_index.device), (node_num,node_num))).to_dense()
    edge_dense = (edge_dense + edge_dense.transpose(-1,-2))/2
    edge_index = edge_dense.nonzero().transpose(-1,-2)

    sym_weight = weight_dense[edge_index[0],edge_index[1]]

    return edge_index, sym_weight

