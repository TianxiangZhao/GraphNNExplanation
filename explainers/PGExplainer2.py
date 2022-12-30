import torch
from torch.cuda import init
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from typing import Optional
from explainers.BaseExplainer import BaseExplainer
from explainers.expl_utils import to_sym
from torch_geometric.loader import DataLoader
import functools
import ipdb

#set masks via multiplying with input
class PGExplainer2(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """
    
    coeffs = {
        'edge_size': 0.05,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, epochs=30, lr=0.003, temp=(5.0, 2.0), sample_bias=0, loss='Tgt', align_emb=False, directional=False, aligner=None, **kwargs):
        super().__init__(model,num_hops, return_type, feat_mask_type, allow_edge_mask,log, loss,directional=directional,align_emb=align_emb, aligner=aligner )

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        
        self.coeffs.update(kwargs)
        
        self.sample_bias = sample_bias
        self.type = kwargs['type']

        if self.type == "graph":
            self.expl_embedding = self.model.nhid * 2*kwargs['nlayer']
        else:
            self.expl_embedding = self.model.nhid * 3*kwargs['nlayer'] #center node, src/tgt node embedding


        #construct model
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.explainer_model = self.explainer_model.to(next(model.parameters()).device)

    def _create_explainer_input(self, pair, embeds, node_id=-1):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        if self.type == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(sampling_weights.device)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  gate_inputs
        else:
            graph = sampling_weights
        return graph

    def prepare(self, dataset, args):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """

        #self.explainer_model = self.explainer_model.to(dataset[0].x.device)

        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        self.train(dataloader, )

    def train(self, dataloader):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()
        self.model.eval()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        # Start training loop
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            t = temp_schedule(e)

            for data in dataloader:
                data = data.to(next(self.model.parameters()).device)
                if self.type == 'node':
                    self.__clear_masks__()
                    embeds = self.model.embedding(data.x, data.edge_index).detach()
                    losses = None

                    for node_idx in data.expl_mask.nonzero().squeeze():
                        node_idx = node_idx.item()
                        kwargs = {}
                        kwargs['embed'] = embeds
                        kwargs['y'] = data.y
                        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
                            self.__subgraph__(node_idx, data.x, data.edge_index, **kwargs)

                        # Sample possible explanation
                        input_expl = self._create_explainer_input(edge_index, kwargs['embed'], mapping).unsqueeze(0)
                        sampling_weights = self.explainer_model(input_expl)
                        if not self.directional:
                            edge_index, sampling_weights = to_sym(edge_index,sampling_weights, x.shape[0])
                        mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                        #get groundtruth
                        self.__clear_masks__()
                        with torch.no_grad():
                            out = self.model(x, edge_index)
                            if self.return_type == 'regression':
                                prediction = out
                            else:
                                pred_logits = self.__to_log_prob__(out)
                        # get embedding for alignment
                        if self.align_emb:
                            embeds_tgt, grads_tgt = self.aligner.get_emb(self.model,x, kwargs['y'], edge_index, mapping)

                        #get explanation loss           
                        self.edge_mask = mask
                        out = self.model(x, edge_index, mask.sigmoid())
                        if self.return_type == 'regression':
                            loss = self.__loss__(mapping, out, prediction)
                        else:
                            log_logits = self.__to_log_prob__(out)
                            loss = self.__loss__(mapping, log_logits, pred_logits)
                        #print('behavior consistency loss: {}'.format(loss.item()))
    
                        if self.align_emb:
                            embeds_src, grads_src = self.aligner.get_emb(self.model, x, kwargs['y'], edge_index, mapping, edge_weight=mask.sigmoid())
                            loss_align, loss_align_list = self.aligner.align_loss(embeds_src,embeds_tgt,mapping,grads_src,grads_tgt)
                            #print('layer-wise alignment loss: {}'.format(loss_align_list))
                            loss = loss+loss_align

                        if losses is None:
                            losses = loss
                        else:
                            losses += loss 
                             
                elif self.type == 'graph':
                    self.__clear_masks__()
                    if data.expl_mask.nonzero().shape[0]>=1:
                        data = Batch.from_data_list(data.index_select(data.expl_mask.nonzero().squeeze()))
                    else:
                        continue

                    embeds = self.model.embedding(data.x, data.edge_index, batch=data.batch).detach()
                    losses = None
                    if True:
                        # Sample possible explanation
                        input_expl = self._create_explainer_input(data.edge_index, embeds).unsqueeze(0)
                        sampling_weights = self.explainer_model(input_expl)
                        if not self.directional:
                            edge_index, sampling_weights = to_sym(data.edge_index,sampling_weights, data.x.shape[0])
                        else:
                            edge_index = data.edge_index
                        mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                        #get groundtruth
                        self.__clear_masks__()
                        with torch.no_grad():
                            out = self.model(data.x, edge_index, batch=data.batch)
                            if self.return_type == 'regression':
                                prediction = out
                            else:
                                pred_logits = self.__to_log_prob__(out)                        
                        # get embedding for alignment
                        if self.align_emb:
                            embeds_tgt, grads_tgt = self.aligner.get_emb(self.model, data.x, data.y, edge_index, -1,batch=data.batch)

                        #get explanation loss           
                        self.edge_mask = mask
                        out = self.model(data.x, edge_index, mask.sigmoid(), batch=data.batch)
                        if self.return_type == 'regression':
                            loss = self.__loss__(-1, out, prediction)
                        else:
                            log_logits = self.__to_log_prob__(out)
                            loss = self.__loss__(-1, log_logits, pred_logits)    
                        #print('behavior consistency loss: {}'.format(loss.item()))
    
                        if self.align_emb:
                            embeds_src, grads_src = self.aligner.get_emb(self.model, data.x, data.y, edge_index, -1, edge_weight=mask.sigmoid(), batch=data.batch)
                            loss_align, loss_align_list = self.aligner.align_loss(embeds_src,embeds_tgt,-1,grads_src,grads_tgt)
                            #print('layer-wise alignment loss: {}'.format(loss_align_list))
                            loss = loss+loss_align

                        if losses is None:
                            losses = loss
                        else:
                            losses += loss 

            losses.backward()
            optimizer.step()


    def explain_graph(self, x, edge_index, batch, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        embeds = self.model.embedding(x, edge_index, batch=batch).detach()
        pred_y = self.model(x, edge_index, batch=batch).detach().argmax(dim=-1)

        # Only operate on a k-hop subgraph around `node_idx`.
        if kwargs is None:
            kwargs = {}
        kwargs['embed'] = embeds
        kwargs['pred_y'] = pred_y

        # Sample possible explanation
        input_expl = self._create_explainer_input(edge_index, kwargs['embed'], -1).unsqueeze(0)
        sampling_weights = self.explainer_model(input_expl)
        if not self.directional and 'edge_label' in kwargs.keys():
            original_edge_index = edge_index
            edge_index, edge_mask = to_sym(original_edge_index,sampling_weights, x.shape[0])
            _,kwargs['edge_label'] = to_sym(original_edge_index,kwargs['edge_label'], x.shape[0])
        edge_mask = torch.sigmoid(self._sample_graph(sampling_weights, training=False)).squeeze()

        
        if hasattr(self, 'node_feat_mask') and self.node_feat_mask is not None:  
            node_feat_mask = self.node_feat_mask.detach().sigmoid()
        else:
            node_feat_mask = None

        return node_feat_mask, edge_mask, edge_index,-1, kwargs


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

        embeds = self.model.embedding(x, edge_index).detach()
        pred_y = self.model(x, edge_index).detach().argmax(dim=-1)

        # Only operate on a k-hop subgraph around `node_idx`.
        if kwargs is None:
            kwargs = {}
        kwargs['embed'] = embeds
        kwargs['pred_y'] = pred_y

        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
            self.__subgraph__(node_idx, x, edge_index, **kwargs)

        # Sample possible explanation
        input_expl = self._create_explainer_input(edge_index, kwargs['embed'], mapping).unsqueeze(0)
        sampling_weights = self.explainer_model(input_expl)
        if not self.directional:
            original_edge_index = edge_index
            edge_index, edge_mask = to_sym(original_edge_index,sampling_weights, x.shape[0])
            _,kwargs['edge_label'] = to_sym(original_edge_index,kwargs['edge_label'], x.shape[0])
        edge_mask = torch.sigmoid(self._sample_graph(sampling_weights, training=False)).squeeze()

        
        if hasattr(self, 'node_feat_mask') and self.node_feat_mask is not None:  
            node_feat_mask = self.node_feat_mask.detach().sigmoid()
        else:
            node_feat_mask = None

        return node_feat_mask, edge_mask, edge_index, mapping.item(), kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}()'