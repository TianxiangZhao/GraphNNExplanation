from math import sqrt
from networkx.algorithms.dag import all_topological_sorts

import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from explainers.BaseExplainer import BaseExplainer
from explainers.expl_utils import to_sym
from torch_geometric.utils import k_hop_subgraph, to_networkx
from typing import Optional
from . import EmbAligner as aligners

import ipdb



class GNNExplainer(BaseExplainer):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
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
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """
    
    coeffs = {
        'edge_size': 0.01,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, loss='Tgt', directional=False, align_emb = False, aligner=None, **kwargs):
        super().__init__(model, num_hops, return_type, feat_mask_type, allow_edge_mask,log,loss,directional=directional, align_emb=align_emb, aligner=aligner)
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.type = kwargs['type']
        
        self.coeffs.update(kwargs)

    def prepare(self, dataset, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain_graph(self, x, edge_index, batch, **kwargs):
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
        self.model.eval()
        self.__clear_masks__()

        num_nodes = x.size(0)

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(x, edge_index, batch=batch)
            if self.return_type == 'regression':
                prediction = out
            else:
                pred_logits = self.__to_log_prob__(out)
        if kwargs is None:
            kwargs = {}
        pred_y = pred_logits.argmax(dim=-1)
        kwargs['pred_y'] = pred_y

        # get embedding for alignment
        if self.align_emb:
            embeds_tgt, grads_tgt = self.aligner.get_emb(self.model,x, kwargs['y'], edge_index, tgt_node=-1, batch=batch)

        #start searching for explanations
        self.__set_masks__(x, edge_index)

        if self.allow_edge_mask:
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain a graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            ##print('feature mask: {}'.format(self.node_feat_mask))
            #h = x * self.node_feat_mask.sigmoid() #not using node feature mask
            h = x
            out = self.model(h, edge_index, batch=batch)
            if self.return_type == 'regression':
                loss = self.__loss__(-1, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(-1, log_logits, pred_logits)
            
            #print('behavior consistency loss: {}'.format(loss.item()))
            if self.align_emb:
                embeds_src, grads_src = self.aligner.get_emb(self.model, h, kwargs['y'], edge_index, tgt_node=-1, batch=batch)
                for emb in embeds_src:
                    if emb.isnan().any():
                        ipdb.set_trace()

                loss_align, loss_align_list = self.aligner.align_loss(embeds_src,embeds_tgt,-1, grads_src,grads_tgt)
                #print('layer-wise alignment loss: {}'.format(loss_align_list))
                loss = loss+loss_align

            loss.backward()
            optimizer.step()
            
            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type == 'individual_feature':
            new_mask = x.new_zeros(num_nodes, x.size(-1))
            new_mask[:] = node_feat_mask
            node_feat_mask = new_mask
        elif self.feat_mask_type == 'scalar':
            new_mask = x.new_zeros(num_nodes, 1)
            new_mask[:] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()

        if not self.directional:
            original_edge_index = edge_index
            edge_index, edge_mask = to_sym(original_edge_index,self.edge_mask.detach(), x.shape[0])
            _,kwargs['edge_label'] = to_sym(original_edge_index,kwargs['edge_label'], x.shape[0])

        else: 
            edge_mask = self.edge_mask.detach()
        edge_mask = edge_mask.sigmoid()
        
        self.__clear_masks__()

        return node_feat_mask, edge_mask, edge_index, -1, kwargs


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

        self.model.eval()
        self.__clear_masks__()

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
            self.__subgraph__(node_idx, x, edge_index, **kwargs)

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(x, edge_index)
            if self.return_type == 'regression':
                prediction = out
            else:
                pred_logits = self.__to_log_prob__(out)
        if kwargs is None:
            kwargs = {}
        pred_y = pred_logits.argmax(dim=-1)
        kwargs['pred_y'] = pred_y

        # get embedding for alignment
        if self.align_emb:
            embeds_tgt, grads_tgt = self.aligner.get_emb(self.model,x, kwargs['y'], edge_index, mapping)

        #start searching for explanations
        self.__set_masks__(x, edge_index)

        if self.allow_edge_mask:
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            ##print('feature mask: {}'.format(self.node_feat_mask))
            #h = x * self.node_feat_mask.sigmoid() #not using node feature mask
            h = x
            out = self.model(h, edge_index)
            if self.return_type == 'regression':
                loss = self.__loss__(mapping, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(mapping, log_logits, pred_logits)
            
            #print('behavior consistency loss: {}'.format(loss.item()))
            if self.align_emb:
                embeds_src, grads_src = self.aligner.get_emb(self.model, h, kwargs['y'], edge_index, mapping)
                for emb in embeds_src:
                    if emb.isnan().any():
                        ipdb.set_trace()

                loss_align, loss_align_list = self.aligner.align_loss(embeds_src,embeds_tgt,mapping, grads_src,grads_tgt)
                #print('layer-wise alignment loss: {}'.format(loss_align_list))
                
                loss = loss+loss_align
            loss.backward()

            optimizer.step()
            

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type == 'individual_feature':
            new_mask = x.new_zeros(num_nodes, x.size(-1))
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        elif self.feat_mask_type == 'scalar':
            new_mask = x.new_zeros(num_nodes, 1)
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()

        if not self.directional:
            original_edge_index = edge_index
            edge_index, edge_mask = to_sym(original_edge_index,self.edge_mask.detach(), x.shape[0])
            _,kwargs['edge_label'] = to_sym(original_edge_index,kwargs['edge_label'], x.shape[0])
        else: 
            edge_mask = self.edge_mask.detach()
        edge_mask = edge_mask.sigmoid()
        
        self.__clear_masks__()


        return node_feat_mask, edge_mask, edge_index, mapping.item(), kwargs



    def __repr__(self):
        return f'{self.__class__.__name__}()'

