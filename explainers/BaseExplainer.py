from abc import ABC, abstractmethod
from math import sqrt
from sklearn import utils

import torch
from torch.cuda import init
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

import explainers.expl_utils as expl_utils
from torch_geometric.utils import k_hop_subgraph, to_networkx
from typing import Optional
import ipdb
import numpy as np
import utils

EPS = 1e-15

class BaseExplainer(ABC):
    r"""Adapted from GNNExplainer implemented in Torch_geometric.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        num_hops (int, optional): The number of hops the :obj:`model` is
            aggregating information from.
            If set to :obj:`None`, will automatically try to detect this
            information based on the number of
            :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`
            layers inside :obj:`model`. (default: :obj:`None`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns raw
            scores) and :obj:`"regression"` (the model returns scalars).
            (default: :obj:`"log_prob"`)
        feat_mask_type (str, optional): Denotes the type of feature mask
            that will be learned. Valid inputs are :obj:`"feature"` (a single
            feature-level mask for all nodes), :obj:`"individual_feature"`
            (individual feature-level masks for each node), and :obj:`"scalar"`
            (scalar mask for each each node). (default: :obj:`"feature"`)
        allow_edge_mask (boolean, optional): If set to :obj:`False`, the edge
            mask will not be optimized. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
        loss (string): choices: {'Tgt', 'Dif', 'Entropy'}
        align_emb: whether aligning embedding in obtain explanation
        aligner: aligners for aligning embeddings
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """

    def __init__(self, model,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, loss='Tgt', directional=False, align_emb=False, aligner=None):
        super().__init__()
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        self.model = model
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.loss = loss
        self.directional = directional
        self.align_emb = align_emb
        
        if self.align_emb:
            assert aligner is not None
            self.aligner = aligner


    def __set_masks__(self, x, edge_index, init=None):

        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        if init is None or 'node_feat_mask' not in init.keys():
            if self.feat_mask_type == 'individual_feature':
                self.node_feat_mask = torch.nn.Parameter((torch.randn(N, F) * std).to(x.device))
            elif self.feat_mask_type == 'scalar':
                self.node_feat_mask = torch.nn.Parameter((torch.randn(N, 1) * std).to(x.device))
            else:
                self.node_feat_mask = torch.nn.Parameter((torch.randn(1, F) * std).to(x.device))
        else:
            self.node_feat_mask = init['node_feat_mask']

        if init is None or 'edge_mask' not in init.keys():
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = torch.nn.Parameter((torch.randn(E) * std).to(x.device))

            if not self.allow_edge_mask:
                self.edge_mask.requires_grad_(False)
                self.edge_mask.fill_(float('inf'))  # `sigmoid()` returns `1`.
            self.loop_mask = edge_index[0] != edge_index[1]
        else:
            self.edge_mask = init['edge_mask']
            if not self.allow_edge_mask:
                self.edge_mask.requires_grad_(False)
                self.edge_mask.fill_(float('inf'))  # `sigmoid()` returns `1`.
            self.loop_mask = edge_index[0] != edge_index[1]

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
                module.__loop_mask__ = self.loop_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
                module.__loop_mask__ = None
        self.node_feat_mask = None
        self.edge_mask = None
        module.loop_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __loss__(self, node_idx, log_logits, pred_logits, loss_type=None):
        # node_idx is -1 for explaining graphs
        if loss_type is None:
            loss_type = self.loss
        if self.return_type == 'regression':
            if node_idx != -1:
                loss = torch.cdist(log_logits[node_idx], pred_logits[node_idx])
            else:
                loss = torch.cdist(log_logits, pred_logits)
        elif loss_type == 'Entropy':
            #calculate cross entropy
            if node_idx != -1:
                loss = -torch.sum(torch.exp(log_logits[node_idx])*pred_logits[node_idx])
            else:
                loss = -torch.sum(torch.exp(log_logits[0])*pred_logits[0])
        elif loss_type == 'Dif':
            if node_idx != -1:
                loss = torch.sum(torch.abs(log_logits[node_idx]-pred_logits[node_idx]))
            else:
                loss = torch.sum(torch.abs(log_logits[0]-pred_logits[0]))
        elif loss_type == 'Tgt':
            pred_label = pred_logits.argmax(dim=-1)
            if node_idx != -1:
                loss = -log_logits[node_idx, pred_label[node_idx]]
            else:
                loss = -log_logits[0, pred_label[0]]

        if hasattr(self, 'edge_mask'):  
            m = self.edge_mask.sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if hasattr(self, 'node_feat_mask') and self.node_feat_mask is not None:        
            m = self.node_feat_mask.sigmoid()
            node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'


    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, subset, kwargs


    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    @abstractmethod
    def prepare(self, dataset, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass


    @abstractmethod
    def explain_graph(self, x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        pass


    @abstractmethod
    def explain_node(self, node_idx,x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        pass
    
    '''
    def fidelity_remove(self, x, edge_index, edge_mask, node_idx=-1, k=7, y_gt=None):
        #gradually remove edges
        agreement_scores = []

        #get original output
        with torch.no_grad():
            if node_idx == -1:
                batch = x.new(x.shape[0]).long().fill_(0)
                self.__clear_masks__()
                out = self.model(x, edge_index, batch=batch)
            else:
                out = self.model(x, edge_index)
            pred_label = out.argmax(dim=-1)

        #get dictionary: src_node, tgt_node -> edge idx, and triangular edges
        idx_dict = {}
        tri_edge_index = edge_index.new(2,edge_index.shape[1]).fill_(0)
        tri_edge_mask = edge_mask.new(edge_mask.shape[0]).fill_(-1)
        passed=0
        for i, (u,v) in enumerate(edge_index.transpose(-1,-2).cpu().numpy()):
            idx_dict[(v,u)] = i
            if (u,v) not in idx_dict.keys():
                tri_edge_index[0,passed] = u
                tri_edge_index[1,passed] = v
                tri_edge_mask[passed] = edge_mask[i]
                passed+=1

        #start testing
        for size in range(k):
            if size > edge_mask.shape[0]:
                size = edge_mask.shape[0]
            if self.directional:
                top_index = edge_mask.topk(k=size)[1]
                new_mask = edge_mask.new(edge_mask.shape).fill_(1)
                new_mask[top_index] = 0
            else:
                if size+1 > passed:
                    size = passed-1
                top_index = tri_edge_mask.topk(k=size)[1]

                new_mask = edge_mask.new(edge_mask.shape).fill_(1)
                for top_i in top_index:
                    u,v = tri_edge_index[:,top_i]
                    new_mask[idx_dict[(u.cpu().item(),v.cpu().item())]] = 0
                    new_mask[idx_dict[(v.cpu().item(),u.cpu().item())]] = 0

            self.__clear_masks__()
            init = {}
            init['edge_mask'] = torch.nn.Parameter(new_mask)
            self.__set_masks__(x,edge_index,init=init)

            if node_idx == -1 or self.type=='graph':
                out = self.model(x, edge_index,batch=batch)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[0]==pred_label[0])
            else:
                out = self.model(x, edge_index)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[node_idx]==pred_label[node_idx])

            
            agreement_scores.append(agreement_score)
        
        self.__clear_masks__()

        return agreement_scores

    def fidelity_add(self,  x, edge_index, edge_mask, node_idx=-1, k=7, y_gt=None):
        
        #gradually add edges
        agreement_scores = []
        
        #get original output
        with torch.no_grad():
            if node_idx == -1:
                batch = x.new(x.shape[0]).long().fill_(0)
                self.__clear_masks__()
                out = self.model(x, edge_index, batch=batch)
            else:
                out = self.model(x, edge_index)
            pred_label = out.argmax(dim=-1)

        #get dictionary: src_node, tgt_node -> edge idx, and triangular edges for undirectional case
        #in undirectional case, require edge_index and edge_mask to be symmetric
        idx_dict = {}
        tri_edge_index = edge_index.new(2,edge_index.shape[1]).fill_(0)
        tri_edge_mask = edge_mask.new(edge_mask.shape[0]).fill_(-1)
        passed=0
        for i, (u,v) in enumerate(edge_index.transpose(-1,-2).cpu().numpy()):
            idx_dict[(v,u)] = i
            if (u,v) not in idx_dict.keys():
                tri_edge_index[0,passed] = u
                tri_edge_index[1,passed] = v
                tri_edge_mask[passed] = edge_mask[i]
                passed+=1

        #start testing
        for size in range(k):
            if size >= edge_mask.shape[0]:
                size = edge_mask.shape[0]
            if self.directional:
                top_index = edge_mask.topk(k=size)[1]
                new_mask = edge_mask.new(edge_mask.shape).fill_(0)
                new_mask[top_index] = 1
            else:
                if size+1 > passed:
                    size = passed-1
                top_index = tri_edge_mask.topk(k=size)[1]

                new_mask = edge_mask.new(edge_mask.shape).fill_(0)
                for top_i in top_index:
                    u,v = tri_edge_index[:,top_i]
                    new_mask[idx_dict[(u.cpu().item(),v.cpu().item())]] = 1
                    new_mask[idx_dict[(v.cpu().item(),u.cpu().item())]] = 1

            self.__clear_masks__()
            init = {}
            init['edge_mask'] = torch.nn.Parameter(new_mask)
            self.__set_masks__(x,edge_index,init=init)
            

            if node_idx == -1 or self.type=='graph':
                out = self.model(x, edge_index, batch=batch)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[0]==pred_label[0])
            else:        
                out = self.model(x, edge_index)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[node_idx]==pred_label[node_idx])

            agreement_scores.append(agreement_score)

        self.__clear_masks__()

        return agreement_scores
        '''

    def fidelity_remove(self, x, edge_index, edge_mask, node_idx=-1, k=7, y_gt=None):
        #gradually remove edges
        agreement_scores = []

        #get original output
        with torch.no_grad():
            if node_idx == -1:
                batch = x.new(x.shape[0]).long().fill_(0)
                self.__clear_masks__()
                out = self.model(x, edge_index, batch=batch)
            else:
                out = self.model(x, edge_index)
            pred_label = out.argmax(dim=-1)

        #get dictionary: src_node, tgt_node -> edge idx, and triangular edges
        idx_dict = {}
        tri_edge_index = edge_index.new(2,edge_index.shape[1]).fill_(0)
        tri_edge_mask = edge_mask.new(edge_mask.shape[0]).fill_(-1)
        passed=0
        for i, (u,v) in enumerate(edge_index.transpose(-1,-2).cpu().numpy()):
            idx_dict[(v,u)] = i
            if (u,v) not in idx_dict.keys():
                tri_edge_index[0,passed] = u
                tri_edge_index[1,passed] = v
                tri_edge_mask[passed] = edge_mask[i]
                passed+=1

        #start testing
        for size in range(k):
            if size >= edge_mask.shape[0]:
                size = edge_mask.shape[0]
            if self.directional:
                top_index = edge_mask.topk(k=size)[1]
                new_mask = edge_mask.new(edge_mask.shape).fill_(1)
                new_mask[top_index] = 0
            else:
                if size+1 > passed:
                    size = passed-1
                top_index = tri_edge_mask.topk(k=size)[1]

                new_mask = edge_mask.new(edge_mask.shape).fill_(1)
                for top_i in top_index:
                    u,v = tri_edge_index[:,top_i]
                    new_mask[idx_dict[(u.cpu().item(),v.cpu().item())]] = 0
                    new_mask[idx_dict[(v.cpu().item(),u.cpu().item())]] = 0

            edge_index_new = edge_index.transpose(-1,0)[new_mask.bool()].transpose(-1,0)

            if node_idx == -1 or self.type=='graph':
                out = self.model(x, edge_index_new,batch=batch)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[0]==pred_label[0])
            else:
                out = self.model(x, edge_index_new)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[node_idx]==pred_label[node_idx])

            
            agreement_scores.append(agreement_score)
        
        self.__clear_masks__()

        return agreement_scores

    def fidelity_add(self,  x, edge_index, edge_mask, node_idx=-1, k=7, y_gt=None):
        
        #gradually add edges
        agreement_scores = []
        
        #get original output
        with torch.no_grad():
            if node_idx == -1:
                batch = x.new(x.shape[0]).long().fill_(0)
                self.__clear_masks__()
                out = self.model(x, edge_index, batch=batch)
            else:
                out = self.model(x, edge_index)
            pred_label = out.argmax(dim=-1)

        #get dictionary: src_node, tgt_node -> edge idx, and triangular edges for undirectional case
        #in undirectional case, require edge_index and edge_mask to be symmetric
        idx_dict = {}
        tri_edge_index = edge_index.new(2,edge_index.shape[1]).fill_(0)
        tri_edge_mask = edge_mask.new(edge_mask.shape[0]).fill_(-1)
        passed=0
        for i, (u,v) in enumerate(edge_index.transpose(-1,-2).cpu().numpy()):
            idx_dict[(v,u)] = i
            if (u,v) not in idx_dict.keys():
                tri_edge_index[0,passed] = u
                tri_edge_index[1,passed] = v
                tri_edge_mask[passed] = edge_mask[i]
                passed+=1

        #start testing
        for size in range(k):
            if size >= edge_mask.shape[0]:
                size = edge_mask.shape[0]
            if self.directional:
                top_index = edge_mask.topk(k=size)[1]
                new_mask = edge_mask.new(edge_mask.shape).fill_(0)
                new_mask[top_index] = 1
            else:
                if size+1 > passed:
                    size = passed-1
                top_index = tri_edge_mask.topk(k=size)[1]

                new_mask = edge_mask.new(edge_mask.shape).fill_(0)
                for top_i in top_index:
                    u,v = tri_edge_index[:,top_i]
                    new_mask[idx_dict[(u.cpu().item(),v.cpu().item())]] = 1
                    new_mask[idx_dict[(v.cpu().item(),u.cpu().item())]] = 1
            
            edge_index_new = edge_index.transpose(-1,0)[new_mask.bool()].transpose(-1,0)
            

            if node_idx == -1 or self.type=='graph':
                out = self.model(x, edge_index_new, batch=batch)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[0]==pred_label[0])
            else:        
                out = self.model(x, edge_index_new)
                log_pred = out.argmax(dim=-1)
                agreement_score = int(log_pred[node_idx]==pred_label[node_idx])

            agreement_scores.append(agreement_score)

        self.__clear_masks__()

        return agreement_scores



    def fidelity_analysis(self, expl_list, k=7, with_std=False):
        # pos_results: gradually adding top edges;
        # neg_results: gradually removing top edges

        tot_num = len(expl_list)
        pos_results = np.zeros(k)
        neg_results = np.zeros(k)

        for gid,expl in enumerate(expl_list):
            pos_result = self.fidelity_add(expl['x'], expl['edge_index'], expl['edge_mask'], expl['center_node'], k=k, y_gt=expl['y_gt'])
            neg_result = self.fidelity_remove(expl['x'], expl['edge_index'], expl['edge_mask'], expl['center_node'], k=k, y_gt=expl['y_gt'])

            for i in range(k):
                pos_results[i] += pos_result[i]
                neg_results[i] += neg_result[i]

        avg_pos = pos_results/tot_num
        avg_neg = neg_results/tot_num

        if with_std:
            #cal std
            pos_std = np.zeros(k)
            neg_std = np.zeros(k)
            for expl in expl_list:
                pos_result = self.fidelity_add(expl['x'], expl['edge_index'], expl['edge_mask'], expl['center_node'], k=k, y_gt=expl['y_gt'])
                neg_result = self.fidelity_remove(expl['x'], expl['edge_index'], expl['edge_mask'], expl['center_node'], k=k, y_gt=expl['y_gt'])

                for i in range(k):
                    pos_std[i] += (pos_result[i]-avg_pos[i])**2
                    neg_std[i] += (neg_result[i]-avg_neg[i])**2

            #pos_std = np.sqrt(pos_std/tot_num)
            #neg_std = np.sqrt(neg_std/tot_num)
            pos_std = pos_std/tot_num
            neg_std = neg_std/tot_num
            
            return avg_pos, avg_neg, pos_std, neg_std

        return avg_pos, avg_neg



    def embed_distance_analysis(self, expl_list):
        ipdb.set_trace()


        return embed_distance_list, agreements

    def stability_analysis(self,expl_lists, Edge_k=5):
        '''
        expl_lists: [[expl, expl,...], [expl, expl, ...], ...], of size: instance_num*explainer_num
        Edge_k: topk explained edges to examine
        return:
            dif_list: [dif, dif, ....] of size: instance_num, returning the average pair-wise SHD edge difference for each instance
            difs: returning the average SHD differnece across all explanations
            diss: returning the average explanation weight distance across all explanations
        '''
        expl_num = len(expl_lists[0])
        dif_list = []
        difs=utils.meters()
        diss=utils.meters()
        for expl_list in expl_lists:
            dif = utils.meters()
            dis = utils.meters()
            for result1 in range(expl_num-1):
                for result2 in range(result1+1, expl_num):
                    mask1 = expl_list[result1].new(expl_list[result1].shape).fill_(0)
                    mask2 = expl_list[result2].new(expl_list[result2].shape).fill_(0)
                    if Edge_k >= expl_list[result1].shape[0]:
                        Edge_k = expl_list[result1].shape[0]

                    index1 = expl_list[result1].topk(k=Edge_k)[1]
                    index2 = expl_list[result2].topk(k=Edge_k)[1]

                    mask1[index1] = expl_list[result1][index1]
                    mask2[index2] = expl_list[result2][index2]

                    dis.update(torch.sum(torch.abs(mask1-mask2)).cpu().item())

                    mask1.fill_(0)
                    mask2.fill_(0)
                    mask1[index1] = 1
                    mask2[index2] = 1

                    diff = torch.sum(torch.abs(mask1-mask2))
                    dif.update(diff.cpu().item())
            dif_list.append(dif.avg())
            difs.update(dif.avg())
            diss.update(dis.avg())
        
        return dif_list, difs.avg(), diss.avg()




    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, edge_y=None, node_alpha=None,
                           node_depict=None, seed=10, **kwargs):
        r"""Visualizes the subgraph given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`-1` to explain graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node.
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from inspect import signature

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        #start drawing
        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        if edge_y is None:
            edge_color = ['black'] * edge_index.size(1)
        else:
            colors = list(plt.rcParams['axes.prop_cycle'])
            edge_color = [
                colors[i % len(colors)]['color']
                for i in edge_y[hard_edge_mask]
            ]

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'],
                        edge_attrs=['att', 'edge_color'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G, seed=seed)        
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()

        if self.directional:
            for source, target, data in G.edges(data=True):
                ax.annotate(
                    '', xy=pos[target], xycoords='data', xytext=pos[source],
                    textcoords='data', arrowprops=dict(
                        arrowstyle="->",
                        alpha=max(data['att'], 0.1),
                        color=data['edge_color'],
                        shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                        shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                        connectionstyle="arc3,rad=0.1",
                    ))
        else:
            G = G.to_undirected()
            exp_edge_list=[]
            non_edge_list=[]
            for edge in G.edges():
                if G[edge[0]][edge[1]]['att'] > 0.1:
                    exp_edge_list.append(edge)
                else:
                    non_edge_list.append(edge)
            nx.draw_networkx_edges(G, pos=pos,edgelist=exp_edge_list, edge_color=data['edge_color'],  alpha=1)
            nx.draw_networkx_edges(G, pos=pos,edgelist=non_edge_list,edge_color=data['edge_color'], alpha=0.1)

        if node_alpha is None:
            nx.draw_networkx_nodes(G, pos, node_color=y.tolist(),
                                   **node_kwargs)
        else:
            node_alpha_subset = node_alpha[subset]
            assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
            nx.draw_networkx_nodes(G, pos, alpha=node_alpha_subset.tolist(),
                                   node_color=y.tolist(), **node_kwargs)

        nx.draw_networkx_labels(G, pos, **label_kwargs)

        if node_depict is not None:
            for ind, word in enumerate(node_depict):
                x,y = pos[ind]
                plt.text(x,y+0.1,s=word, bbox=dict(facecolor='red', alpha=0.5),horizontalalignment='center')

        return fig, G


    def __repr__(self):
        return f'{self.__class__.__name__}()'