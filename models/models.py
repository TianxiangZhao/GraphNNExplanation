import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv
import ipdb
from torch_geometric.nn import global_mean_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(GCNConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, return_logits=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if return_logits:
            if self.res:
                x = torch.cat(xs, dim=-1)
            y = self.lin(x)

            if return_list:
                return xs, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)

        return x

class GraphGCN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(GCNConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer*2, nclass)
        else:
            self.lin = Linear(nhid*2, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None: # No batch given
            print('no batch info given')
            ipdb.set_trace()
            batch = x.new(x.size(0)).long().fill_(0)

        x = self.embedding(x,edge_index,edge_weight, batch=batch, graph_level=True)# 

        x = self.lin(x)
        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, batch=None, return_list=False, return_logits=False, graph_level=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if self.res:
            x = torch.cat(xs, dim=-1)

        out1 = global_max_pool(x, batch)
        out2 = global_mean_pool(x, batch)
        
        if graph_level:
            emb_list = [out1, out2]
            x = torch.cat([out1,out2],dim=-1)
        else:
            emb_list = xs

        if return_logits:
            y = self.lin(torch.cat([out1,out2],dim=-1))

            if return_list:
                return emb_list, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return emb_list
            else:
                return x