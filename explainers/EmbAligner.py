import torch
import torch.nn.functional as F
import ipdb
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from torch.optim import Adam
from tqdm import tqdm

class EmbAligner(object):
    """
    measure the alignment w.r.t intermediate embeddings
    Args:
        loss: the criterion used for computing the gradients on embeddings
        need_grad: whether use gradients in calculating alignment loss
    """
    def __init__(self, loss='nll', need_grad=True, align_weight=1.0):

        if loss == 'nll':
            self.loss = torch.nn.NLLLoss()

        self.need_grad = need_grad
        self.align_weight = align_weight


    def get_emb(self, model, x, y, edge_index, tgt_node=-1, edge_weight = None, batch=None):
        '''
        Args:
            model: GNN model
            tgt_node: the index of node to be explained. Set to -1 for graph classification task
            batch: used for graph classification. 
        Return:
            list of embeddings
            list of gradients on each embedding
        '''
        if tgt_node  != -1:#node classification task
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight)
            for emb in emb_list:
                emb.retain_grad()

            loss = self.loss(logits[tgt_node], y[tgt_node])
            #loss.backward(retain_graph=True, create_graph=True)
            loss.backward(retain_graph=True)

            grad_list = []
            for emb in emb_list:
                grad_list.append(emb.grad)
        else:#graph classification task
            assert batch is not None, "graph classification should provide batch info for obtaining embedding"
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight, batch=batch, graph_level=True)
            for emb in emb_list:
                emb.retain_grad()

            loss = self.loss(logits, y)
            #loss.backward(retain_graph=True, create_graph=True)
            loss.backward(retain_graph=True)

            grad_list = []
            for emb in emb_list:
                grad_list.append(emb.grad)

        normed_grad_list = []
        for grad in grad_list:
            avg = torch.mean(grad)
            std = torch.sqrt(torch.std(grad))
            normed_grad = torch.clamp(grad, min=avg-std, max=avg+std)
            scale = 1/torch.mean(torch.norm(normed_grad.detach(),dim=-1))
            normed_grad = normed_grad*scale
            normed_grad_list.append(normed_grad)


        return emb_list, normed_grad_list

    def align_loss(self, embeds_src, embeds_tgt, node_idx=-1, grads_src=None, grads_tgt=None):
        '''
        calculate alignment loss based on embeddings and grads on them

        Args:
            embeds: list of embeddings
            grads: list of gradients. used if self.need_grad is True
            node_idx: the index of node to-be-aligned. -1 indicates graph classification
        Return:
            loss_a: sum of layer-wise alignment loss
            loss_list: layer-wise alignment loss in the list
        '''
        
        loss_a = None
        loss_list = []

        if node_idx !=-1:
            for layer, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):

                if self.need_grad:
                    emb_src = emb_src*(grads_tgt[layer].detach())
                    emb_tgt = emb_tgt*grads_tgt[layer]

                if loss_a is None:
                    loss_a = F.mse_loss(emb_src[node_idx], emb_tgt[node_idx].detach())
                    loss_list.append(loss_a.detach())
                else:
                    loss = F.mse_loss(emb_src[node_idx], emb_tgt[node_idx].detach())
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()
        else:
            for out, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):
                if self.need_grad:
                    emb_src = emb_src*(grads_tgt[out].detach())
                    emb_tgt = emb_tgt*grads_tgt[out]

                if loss_a is None:
                    loss_a = F.mse_loss(emb_src, emb_tgt.detach())
                    loss_list.append(loss_a.detach())
                else:
                    loss = F.mse_loss(emb_src, emb_tgt.detach())
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()

        return loss_a*self.align_weight, loss_list

        
class AnchorAligner(object):
    """
    measure the alignment w.r.t intermediate embeddings with anchors
    Args:
        loss: the criterion used for computing the gradients on embeddings
        need_grad: whether use gradients in calculating alignment loss
    """
    def __init__(self, loss='nll', need_grad=True, align_weight=1.0):

        if loss == 'nll':
            self.loss = torch.nn.NLLLoss()

        self.need_grad = need_grad
        self.align_weight = align_weight

        self.anchors = None
        self.anchor_grads = None
        self.transform_weights = None #map to anchor space
        self.transform_biases = None
        #self.distance = 'Cos'
        self.distance = 'Euc'

    def get_emb(self, model, x, y, edge_index, tgt_node=-1, edge_weight = None, batch=None):
        '''
        Args:
            model: GNN model
            tgt_node: the index of node to be explained. Set to -1 for graph classification task
            batch: used for graph classification. 
        Return:
            list of embeddings
            list of gradients on each embedding
        '''
        if tgt_node  != -1:#node classification task
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight)
            for emb in emb_list:
                emb.retain_grad()

            loss = self.loss(logits[tgt_node], y[tgt_node])
            #loss.backward(retain_graph=True, create_graph=True)
            loss.backward(retain_graph=True)

            grad_list = []
            for emb in emb_list:
                grad_list.append(emb.grad)
        else:#graph classification task
            assert batch is not None, "graph classification should provide batch info for obtaining embedding"
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight, batch=batch, graph_level=True)
            for emb in emb_list:
                emb.retain_grad()

            loss = self.loss(logits, y)
            #loss.backward(retain_graph=True, create_graph=True)
            loss.backward(retain_graph=True)

            grad_list = []
            for emb in emb_list:
                grad_list.append(emb.grad)

        normed_grad_list = []
        for grad in grad_list:
            avg = torch.mean(grad)
            std = torch.sqrt(torch.std(grad))
            normed_grad = torch.clamp(grad, min=avg-std, max=avg+std)
            scale = 1/torch.mean(torch.norm(normed_grad.detach(),dim=-1))
            normed_grad = normed_grad*scale
            normed_grad_list.append(normed_grad)

        return emb_list, normed_grad_list

    def _transform(self, embedding, layer):
        #transform an original embedding to its representation in the anchor space
        embedding = embedding-self.transform_biases[layer]
        embedding = torch.matmul(embedding, self.transform_weights[layer])

        if self.distance=='Cos':# use cos distance
            distance = [F.cosine_similarity(emb, self.anchors[layer]) for emb in embedding]
            distance = torch.stack(distance)

        elif self.distance=='Euc': # use Euclidean distance
            distance = [F.pairwise_distance(emb, self.anchors[layer]) for emb in embedding]

            distance = torch.stack(distance)
                
        return distance
        


    def align_loss(self, embeds_src, embeds_tgt, node_idx=-1, grads_src=None, grads_tgt=None):
        '''
        calculate alignment loss based on embeddings and grads on them

        Args:
            embeds: list of embeddings
            grads: list of gradients. used if self.need_grad is True
            node_idx: the index of node to-be-aligned. -1 indicates graph classification
        Return:
            loss_a: sum of layer-wise alignment loss
            loss_list: layer-wise alignment loss in the list
        '''
        
        loss_a = None
        loss_list = []

        if node_idx !=-1:
            for layer, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):

                if self.need_grad:
                    emb_src = emb_src*(self.anchor_grads[layer])
                    emb_tgt = emb_tgt*(self.anchor_grads[layer])

                emb_src = self._transform(emb_src[node_idx], layer)
                emb_tgt = self._transform(emb_tgt[node_idx], layer)

                if loss_a is None:
                    loss_a = F.mse_loss(emb_src, emb_tgt.detach())
                    loss_list.append(loss_a.detach())
                else:
                    loss = F.mse_loss(emb_src, emb_tgt.detach())
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()
        else:
            for out, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):

                if self.need_grad:
                    emb_src = emb_src*(self.anchor_grads[out])
                    emb_tgt = emb_tgt*(self.anchor_grads[out])

                emb_src = self._transform(emb_src, out)
                emb_tgt = self._transform(emb_tgt, out)

                if loss_a is None:
                    loss_a = F.mse_loss(emb_src, emb_tgt.detach())
                    loss_list.append(loss_a.detach())
                else:
                    loss = F.mse_loss(emb_src, emb_tgt.detach())
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()

        return loss_a*self.align_weight, loss_list

    def set_anchors(self, dataset, model,args, num_hops=-1):
        #-----------------------------------
        #construct anchors for alignment
        #-----------------------------------

        #initialize
        model.eval()
        if num_hops == -1:
            num_hops = 0
            for module in model.modules():
                if isinstance(module, MessagePassing):
                    num_hops += 1


        #collect list of embeddings
        if args.datatype == 'node':
            pbar = tqdm(total=dataset[0].x.shape[0])
            pbar.set_description(f'obtain embedding for anchors')
            data = dataset[0]
            data = data.to(next(model.parameters()).device)
            embed_lists = [None for i in range(num_hops)] #one list for each layer
            grad_lists = [None for i in range(num_hops)]

            for node_idx in range(data.x.shape[0]):
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, num_hops, data.edge_index, relabel_nodes=True,
                    num_nodes=data.x.shape[0])

                emb, grad = self.get_emb(model, data.x[subset], data.y[subset], edge_index, tgt_node=mapping)
                for layer in range(num_hops):
                    if embed_lists[layer] is None:
                        embed_lists[layer] = emb[layer][mapping].detach()
                        grad_lists[layer] = grad[layer][mapping].detach()
                    else:
                        embed_lists[layer] = torch.cat((embed_lists[layer], emb[layer][mapping].detach()), dim=0)
                        grad_lists[layer] = torch.cat((grad_lists[layer], grad[layer][mapping].detach()), dim=0)
                pbar.update(1)

        elif args.datatype == 'graph':
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            pbar = tqdm(total=len(dataloader))
            pbar.set_description(f'obtain embedding for anchors')
            embed_lists = [None for i in range(2)] #one list for each pooling
            grad_lists = [None for i in range(2)]

            for data in dataloader:            
                data = data.to(next(model.parameters()).device)
                emb, grad = self.get_emb(model, data.x, data.y, data.edge_index, batch=data.batch)
                for layer in range(2):
                    if embed_lists[layer] is None:
                        embed_lists[layer] = emb[layer].detach()
                        grad_lists[layer] = grad[layer].detach()
                    else:
                        embed_lists[layer] = torch.cat((embed_lists[layer], emb[layer].detach()), dim=0)
                        grad_lists[layer] = torch.cat((grad_lists[layer], grad[layer].detach()), dim=0)
                pbar.update(1)
        pbar.close()

        #set anchors
        self.anchors = []
        self.anchor_grads = []
        self.transform_weights = []
        self.transform_biases = []

        for embeds, grads in zip(embed_lists,grad_lists):
            grads = torch.mean(grads, dim=0)
            if self.need_grad:
                embeds = torch.mul(embeds, grads.unsqueeze(0))
            embed_np = embeds.cpu().numpy()

            #transform
            pca = PCA(n_components=4)
            trans_embed_np = pca.fit_transform(embed_np)
            weight_np = pca.components_.T
            bias_np = pca.mean_
            weight =  embeds.new(weight_np)
            bias = embeds.new(bias_np)
            scale = 100/(trans_embed_np.max()+0.000000000000001)
            trans_embed_np = trans_embed_np*scale*0.5
            self.transform_weights.append(weight*scale)
            self.transform_biases.append(bias)

            #get anchors
            anchor_list = []
            eps=3
            if trans_embed_np.shape[0] <=600:
                min_sample = 4
            else:
                min_sample=10
            clust_model = DBSCAN(eps=eps, min_samples=min_sample).fit(trans_embed_np)
            tried_iter = 0
            while len(set(clust_model.labels_)) <=10 or len(set(clust_model.labels_)) >=60:
                if len(set(clust_model.labels_)) <=10 and set(clust_model.labels_) != {-1} :
                    eps = eps*0.9
                else:
                    eps = eps*1.1
                tried_iter +=1            
                clust_model = DBSCAN(eps=eps, min_samples=min_sample).fit(trans_embed_np)
                if tried_iter >= 1000:
                    print("cannot find suitable parameter for DBSCAN in obtaining anchors")
                    ipdb.set_trace()
                    raise Exception('fail to find DBSCAN paramter')
            clus = clust_model.labels_
            for group in set(clus):
                if group != -1:
                    sel_idx = clus==group
                    sel_samples = trans_embed_np[sel_idx]
                    anchor_list.append(sel_samples.mean(axis=0))
            anchor_list = np.stack(anchor_list, axis=0)
            anchor_list = embeds.new(anchor_list)

            self.anchor_grads.append(grads)
            self.anchors.append(anchor_list)

        return


class BothAligner(object):
    """
    measure the alignment w.r.t intermediate embeddings
    Args:
        loss: the criterion used for computing the gradients on embeddings
        need_grad: whether use gradients in calculating alignment loss
    """
    def __init__(self, loss='nll', need_grad=True, align_weight=1.0,aligner_combine_weight=1.0):

        self.aligner1 = EmbAligner(loss,need_grad,align_weight)
        self.aligner2 = AnchorAligner(loss,need_grad,align_weight)
        self.aligner_combine_weight=aligner_combine_weight


    def get_emb(self, model, x, y, edge_index, tgt_node=-1, edge_weight = None, batch=None):
        '''
        Args:
            model: GNN model
            tgt_node: the index of node to be explained. Set to -1 for graph classification task
            batch: used for graph classification. 
        Return:
            list of embeddings
            list of gradients on each embedding
        '''
        return self.aligner1.get_emb(model,x,y,edge_index,tgt_node,edge_weight,batch)


    def align_loss(self, embeds_src, embeds_tgt, node_idx=-1, grads_src=None, grads_tgt=None):
        '''
        calculate alignment loss based on embeddings and grads on them

        Args:
            embeds: list of embeddings
            grads: list of gradients. used if self.need_grad is True
            node_idx: the index of node to-be-aligned. -1 indicates graph classification
        Return:
            loss_a: sum of layer-wise alignment loss
            loss_list: layer-wise alignment loss in the list
        '''
        loss1,loss_list1 = self.aligner1.align_loss(embeds_src,embeds_tgt,node_idx,grads_src,grads_tgt)
        loss2, loss_list2 = self.aligner2.align_loss(embeds_src,embeds_tgt,node_idx,grads_src,grads_tgt)

        return loss1*self.aligner_combine_weight+loss2, [a+b for a,b in zip(loss_list1,loss_list2)]

    def set_anchors(self, dataset, model,args, num_hops=-1):
        self.aligner2.set_anchors(dataset,model,args,num_hops)

        return

class MixGaussAligner(object):
    """
    measure the alignment w.r.t intermediate embeddings with anchors, from Gaussian mixture distribution perspective
    Args:
        loss: the criterion used for computing the gradients on embeddings
        need_grad: whether use gradients in calculating alignment loss
    """
    def __init__(self, loss='nll', need_grad=False, align_weight=1.0):

        if loss == 'nll':
            self.loss = torch.nn.NLLLoss()

        self.need_grad = need_grad
        self.align_weight = align_weight

        self.anchors = None
        self.anchor_grads = None
        self.transform_weights = None #map to anchor space
        self.transform_biases = None
        #self.distance = 'Cos'
        self.distance = 'Euc'


    def get_emb(self, model, x, y, edge_index, tgt_node=-1, edge_weight = None, batch=None):
        '''
        Args:
            model: GNN model
            tgt_node: the index of node to be explained. Set to -1 for graph classification task
            batch: used for graph classification. 
        Return:
            list of embeddings
            list of gradients on each embedding
        '''
        if tgt_node  != -1:#node classification task
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight)
            for emb in emb_list:
                emb.retain_grad()

            loss = self.loss(logits[tgt_node], y[tgt_node])
            #loss.backward(retain_graph=True, create_graph=True)
            loss.backward(retain_graph=True)

            grad_list = []
            for emb in emb_list:
                grad_list.append(emb.grad)
        else:#graph classification task
            assert batch is not None, "graph classification should provide batch info for obtaining embedding"
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight, batch=batch, graph_level=True)
            for emb in emb_list:
                emb.retain_grad()

            loss = self.loss(logits, y)
            #loss.backward(retain_graph=True, create_graph=True)
            loss.backward(retain_graph=True)

            grad_list = []
            for emb in emb_list:
                grad_list.append(emb.grad)

        normed_grad_list = []
        for grad in grad_list:
            avg = torch.mean(grad)
            std = torch.sqrt(torch.std(grad))
            normed_grad = torch.clamp(grad, min=avg-std, max=avg+std)
            scale = 1/torch.mean(torch.norm(normed_grad.detach(),dim=-1))
            normed_grad = normed_grad*scale
            normed_grad_list.append(normed_grad)

        return emb_list, normed_grad_list

    def _transform(self, embedding, layer):
        #transform an original embedding to its representation in the anchor space
        embedding = embedding-self.transform_biases[layer]
        embedding = torch.matmul(embedding, self.transform_weights[layer])

        if self.distance=='Cos':# use cos distance
            distance = [F.cosine_similarity(emb, self.anchors[layer]) for emb in embedding]
            distance = torch.stack(distance)

        elif self.distance=='Euc': # use Euclidean distance
            distance = [F.pairwise_distance(emb, self.anchors[layer]) for emb in embedding]

            distance = torch.stack(distance)

        #    
        min_dis = distance.min().detach()
        p_dist = [torch.exp(-torch.square(dist/min_dis)) for dist in distance]
        distance = torch.stack(p_dist)
        distance = distance/(distance.sum().detach()+0.001)

                
        return distance
        


    def align_loss(self, embeds_src, embeds_tgt, node_idx=-1, grads_src=None, grads_tgt=None):
        '''
        calculate alignment loss based on embeddings and grads on them

        Args:
            embeds: list of embeddings
            grads: list of gradients. used if self.need_grad is True
            node_idx: the index of node to-be-aligned. -1 indicates graph classification
        Return:
            loss_a: sum of layer-wise alignment loss
            loss_list: layer-wise alignment loss in the list
        '''
        
        loss_a = None
        loss_list = []

        if node_idx !=-1:
            for layer, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):

                if self.need_grad:
                    emb_src = emb_src*(self.anchor_grads[layer])
                    emb_tgt = emb_tgt*(self.anchor_grads[layer])

                emb_src = self._transform(emb_src[node_idx], layer)
                emb_tgt = self._transform(emb_tgt[node_idx], layer)

                if loss_a is None:
                    loss_a = F.kl_div(torch.log(emb_src+0.0001), emb_tgt.detach())

                    loss_list.append(loss_a.detach())
                else:
                    loss = F.kl_div(torch.log(emb_src+0.0001), emb_tgt.detach())
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()
        else:
            for out, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):

                if self.need_grad:
                    emb_src = emb_src*(self.anchor_grads[out])
                    emb_tgt = emb_tgt*(self.anchor_grads[out])

                emb_src = self._transform(emb_src, out)
                emb_tgt = self._transform(emb_tgt, out)

                if loss_a is None:
                    loss_a =  F.kl_div(torch.log(emb_src+0.0001), emb_tgt.detach())
                    loss_list.append(loss_a.detach())
                else:
                    loss = F.kl_div(torch.log(emb_src+0.0001), emb_tgt.detach())
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()

        return loss_a*self.align_weight, loss_list

    def set_anchors(self, dataset, model,args, num_hops=-1):
        #-----------------------------------
        #construct anchors for alignment
        #-----------------------------------

        #initialize
        model.eval()
        if num_hops == -1:
            num_hops = 0
            for module in model.modules():
                if isinstance(module, MessagePassing):
                    num_hops += 1


        #collect list of embeddings
        if args.datatype == 'node':
            pbar = tqdm(total=dataset[0].x.shape[0])
            pbar.set_description(f'obtain embedding for anchors')
            data = dataset[0]
            data = data.to(next(model.parameters()).device)
            embed_lists = [None for i in range(num_hops)] #one list for each layer
            grad_lists = [None for i in range(num_hops)]

            for node_idx in range(data.x.shape[0]):
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, num_hops, data.edge_index, relabel_nodes=True,
                    num_nodes=data.x.shape[0])

                emb, grad = self.get_emb(model, data.x[subset], data.y[subset], edge_index, tgt_node=mapping)
                for layer in range(num_hops):
                    if embed_lists[layer] is None:
                        embed_lists[layer] = emb[layer][mapping].detach()
                        grad_lists[layer] = grad[layer][mapping].detach()
                    else:
                        embed_lists[layer] = torch.cat((embed_lists[layer], emb[layer][mapping].detach()), dim=0)
                        grad_lists[layer] = torch.cat((grad_lists[layer], grad[layer][mapping].detach()), dim=0)
                pbar.update(1)

        elif args.datatype == 'graph':
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            pbar = tqdm(total=len(dataloader))
            pbar.set_description(f'obtain embedding for anchors')
            embed_lists = [None for i in range(2)] #one list for each pooling
            grad_lists = [None for i in range(2)]

            for data in dataloader:            
                data = data.to(next(model.parameters()).device)
                emb, grad = self.get_emb(model, data.x, data.y, data.edge_index, batch=data.batch)
                for layer in range(2):
                    if embed_lists[layer] is None:
                        embed_lists[layer] = emb[layer].detach()
                        grad_lists[layer] = grad[layer].detach()
                    else:
                        embed_lists[layer] = torch.cat((embed_lists[layer], emb[layer].detach()), dim=0)
                        grad_lists[layer] = torch.cat((grad_lists[layer], grad[layer].detach()), dim=0)
                pbar.update(1)
        pbar.close()

        #set anchors
        self.anchors = []
        self.anchor_grads = []
        self.transform_weights = []
        self.transform_biases = []

        for embeds, grads in zip(embed_lists,grad_lists):
            grads = torch.mean(grads, dim=0)
            if self.need_grad:
                embeds = torch.mul(embeds, grads.unsqueeze(0))
            embed_np = embeds.cpu().numpy()

            #transform
            pca = PCA(n_components=4)
            trans_embed_np = pca.fit_transform(embed_np)
            weight_np = pca.components_.T
            bias_np = pca.mean_
            weight =  embeds.new(weight_np)
            bias = embeds.new(bias_np)
            scale = 100/(trans_embed_np.max()+0.000000000000001)
            trans_embed_np = trans_embed_np*scale*0.5
            self.transform_weights.append(weight*scale)
            self.transform_biases.append(bias)

            #get anchors
            anchor_list = []
            eps=3
            if trans_embed_np.shape[0] <=600:
                min_sample = 4
            else:
                min_sample=10
            clust_model = DBSCAN(eps=eps, min_samples=min_sample).fit(trans_embed_np)
            tried_iter = 0
            while len(set(clust_model.labels_)) <=10 or len(set(clust_model.labels_)) >=60:
                if len(set(clust_model.labels_)) <=10 and set(clust_model.labels_) != {-1} :
                    eps = eps*0.9
                else:
                    eps = eps*1.1
                tried_iter +=1            
                clust_model = DBSCAN(eps=eps, min_samples=min_sample).fit(trans_embed_np)
                if tried_iter >= 1000:
                    print("cannot find suitable parameter for DBSCAN in obtaining anchors")
                    ipdb.set_trace()
                    raise Exception('fail to find DBSCAN paramter')
            clus = clust_model.labels_
            for group in set(clus):
                if group != -1:
                    sel_idx = clus==group
                    sel_samples = trans_embed_np[sel_idx]
                    anchor_list.append(sel_samples.mean(axis=0))
            anchor_list = np.stack(anchor_list, axis=0)
            anchor_list = embeds.new(anchor_list)

            self.anchor_grads.append(grads)
            self.anchors.append(anchor_list)

        return

class MIAligner(object):
    """
    measure the alignment w.r.t intermediate embeddings with a discriminator
    Args:
        loss: the criterion used for computing the gradients on embeddings
        need_grad: whether use gradients in calculating alignment loss
        layer_id: the layer to be used for MI estimation
    """
    def __init__(self, loss='nll', need_grad=False, align_weight=1.0, layer_id=1, args=None,):

        if loss == 'nll':
            self.loss = torch.nn.NLLLoss()
        assert need_grad is not True, "MIaligner not implemented for need grad"

        self.need_grad = need_grad
        self.align_weight = align_weight
        self.layer_id = layer_id
        self.args=args
        if args.datatype == 'node':
            self.mi_head = torch.nn.Sequential(torch.nn.Linear(args.nhid*2, 32), torch.nn.ReLU(), torch.nn.Linear(32,1)).to(args.device)
        else:
            self.mi_head = torch.nn.Sequential(torch.nn.Linear(args.nhid*2*3, 32), torch.nn.ReLU(), torch.nn.Linear(32,1)).to(args.device)
    
        self.mi_opt = Adam(self.mi_head.parameters(), lr=0.01)


    def get_emb(self, model, x, y, edge_index, tgt_node=-1, edge_weight = None, batch=None, with_random=False):
        '''
        Args:
            model: GNN model
            tgt_node: the index of node to be explained. Set to -1 for graph classification task
            batch: used for graph classification. 
        Return:
            list of embeddings
            list of gradients on each embedding
        '''
        if with_random:
            model.train()
        else:
            model.eval()

        if self.args.datatype == 'node':#node classification task
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight)

        else:#graph classification task
            assert batch is not None, "graph classification should provide batch info for obtaining embedding"
            emb_list, logits = model.embedding(x, edge_index, return_list=True, return_logits=True, edge_weight=edge_weight, batch=batch, graph_level=True)

        normed_grad_list = []


        return emb_list, normed_grad_list

    def align_loss(self, embeds_src, embeds_tgt, node_idx=-1, grads_src=None, grads_tgt=None):
        '''
        calculate alignment loss based on embeddings and grads on them

        Args:
            embeds: list of embeddings
            grads: list of gradients. used if self.need_grad is True
            node_idx: the index of node to-be-aligned. -1 indicates graph classification
        Return:
            loss_a: sum of layer-wise alignment loss
            loss_list: layer-wise alignment loss in the list
        '''
        
        loss_a = None
        loss_list = []

        if node_idx !=-1:
            for layer, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):

                if loss_a is None:
                    loss_a = F.softplus(-self.mi_head(torch.cat([emb_src[node_idx],emb_tgt[node_idx].detach()], dim=-1))).mean()
                    loss_list.append(loss_a.detach())
                else:
                    loss = F.softplus(-self.mi_head(torch.cat([emb_src[node_idx],emb_tgt[node_idx].detach()], dim=-1))).mean()
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()
        else:
            for out, (emb_src, emb_tgt) in enumerate(zip(embeds_src, embeds_tgt)):
                if loss_a is None:
                    loss_a = F.softplus(-self.mi_head(torch.cat([emb_src,emb_tgt.detach()], dim=-1))).mean()
                    loss_list.append(loss_a.detach())
                else:
                    loss = F.softplus(-self.mi_head(torch.cat([emb_src,emb_tgt.detach()], dim=-1))).mean()
                    loss_a = loss_a + loss
                    loss_list.append(loss.detach())
                    
                if loss_a.isnan() or loss_a.isinf():
                    ipdb.set_trace()

        return loss_a*self.align_weight, loss_list

    def collect_embs_all(self, model, dataset, with_random=False, args=None, num_hops=-1):
        
        if num_hops == -1:
            num_hops = 0
            for module in model.modules():
                if isinstance(module, MessagePassing):
                    num_hops += 1
        
        if args.datatype == 'node':
            data = dataset[0]
            data = data.to(next(model.parameters()).device)
            embed_lists = [None for i in range(num_hops)] #one list for each layer
            grad_lists = [None for i in range(num_hops)]

            emb, grad = self.get_emb(model, data.x, data.y, data.edge_index, with_random=with_random)
            for layer in range(num_hops):
                if embed_lists[layer] is None:
                    embed_lists[layer] = emb[layer].detach()
                else:
                    embed_lists[layer] = torch.cat((embed_lists[layer], emb[layer].detach()), dim=0)

        elif args.datatype == 'graph':
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            embed_lists = [None for i in range(2)] #one list for each pooling
            grad_lists = [None for i in range(2)]

            for data in dataloader:            
                data = data.to(next(model.parameters()).device)
                emb, grad = self.get_emb(model, data.x, data.y, data.edge_index, batch=data.batch, with_random=with_random)
                for layer in range(2):
                    if embed_lists[layer] is None:
                        embed_lists[layer] = emb[layer].detach()
                    else:
                        embed_lists[layer] = torch.cat((embed_lists[layer], emb[layer].detach()), dim=0)

        return embed_lists, grad_lists

    def train(self, dataset, model, args, epochs=500):
        self.mi_head.train()

        with torch.no_grad():

            embed_lists, grad_lists = self.collect_embs_all(model, dataset, with_random=False, args=args)
            embed_tgt = embed_lists[self.layer_id]

        for epoch in range(epochs):
            model.dropout=0.1
            embed_lists, grad_lists = self.collect_embs_all(model, dataset, with_random=True, args=args)
            embed_src = embed_lists[self.layer_id]
            print('mi trained for epoch {}'.format(epoch))

            for i_batch in range(100):
                self.mi_opt.zero_grad()

                batch_idx = torch.randint(high=embed_tgt.shape[0], size=(32,)).to(args.device)

                # estimate positive
                pos_MI_score = -F.softplus(-self.mi_head(torch.cat([embed_src[batch_idx],embed_tgt[batch_idx]], dim=-1))).mean()

                # estimate neg
                rand_idx = torch.randint(high=embed_tgt.shape[0], size=(32,)).to(args.device)
                neg_MI_score = F.softplus(self.mi_head(torch.cat([embed_src[batch_idx],embed_tgt[rand_idx]], dim=-1))).mean()

                MI_est = pos_MI_score - neg_MI_score

                MI_loss = -MI_est
                MI_loss.backward()
                self.mi_opt.step()

        return



                



        




        return