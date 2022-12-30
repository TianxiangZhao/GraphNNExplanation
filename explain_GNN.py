from networkx.algorithms.distance_measures import center
import torch
import numpy as np
import random
import os

from torch.nn.functional import threshold
import datasets as datasets
from torch_geometric.loader import DataLoader,RandomNodeSampler
import models.models as models
import utils
import trainers.trainer as trainer
import math
import torch_geometric.datasets as tg_dataset

from tensorboardX import SummaryWriter
from explainers import GNNExplainer, PGExplainer, PGExplainer2
import ipdb
from plots import plot_dist1D, plot_chart
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from explainers import EmbAligner as aligners

###configure arguments
args = utils.get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.task = 'expl'
device = torch.device('cuda' if args.cuda else 'cpu')
args.device=device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.log:
    tb_path = './tensorboard_journal/{}/{}_{}/model{}/edgesize{}ent{}epoch{}exploss{}directional{}align{}aligner{}grad{}weight{}comb{}'.format(args.dataset,args.task, args.explainer, args.model,args.edge_size,args.edge_ent,args.epochs, args.expl_loss,args.directional, args.align_emb,args.aligner, args.align_with_grad, args.align_weight,args.aligner_combine_weight)
    
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    writer = SummaryWriter(tb_path)

###load dataset
if args.datatype == 'node':
    if args.dataset=='BA_shapes':
        dataset = datasets.SyncShapes('BA_shapes')
    elif args.dataset=='Tree_cycle':
        dataset = datasets.SyncShapes('Tree_cycle')
    elif args.dataset=='Tree_grid':
        dataset = datasets.SyncShapes('Tree_grid')
    elif args.dataset=='infected':
        dataset = datasets.infection()
    elif args.dataset=='LoadBA_shapes':
        dataset = datasets.LoadSyn('BA_shapes')
    elif args.dataset=='LoadTree_cycle':
        dataset = datasets.LoadSyn('Tree_cycle')
    elif args.dataset=='LoadTree_grid':
        dataset = datasets.LoadSyn('Tree_grid')
    else:
        ipdb.set_trace()
        print('error, unrecognized node classification dataset, {}'.format(args.dataset))

    args.nfeat = dataset[0].x.shape[-1]
    args.nclass = len(set(dataset[0].y.tolist()))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
elif args.datatype == 'graph':
    if args.dataset == 'mutag':
        dataset = datasets.MoleculeDataset(root='./datasets/', name='MUTAG')
        args.nclass = 2
    elif args.dataset.split('_')[0] == 'SpuriousMotif':
        name, mix_ratio = args.dataset.split('_')
        mix_ratio = float(mix_ratio)
        dataset = datasets.SyncGraphs(name, mix_ratio=mix_ratio)
        dataset = dataset.shuffle()
        args.nclass = 3

    args.nfeat = dataset[0].x.shape[-1]
    graph_num = len(dataset)
    graph_list = []
    for data in dataset:
        if data.expl_mask:
            graph_list.append(data)

###construct model
if args.datatype == 'node':
    if args.model == 'gcn':
        model = models.GCN(args, nfeat=args.nfeat, 
                nhid=args.nhid, 
                nclass=args.nclass, 
                dropout=args.dropout,
                nlayer=args.nlayer)
elif args.datatype == 'graph':
    if args.model == 'gcn':
        model = models.GraphGCN(args, nfeat=args.nfeat, 
                nhid=args.nhid, 
                nclass=args.nclass, 
                dropout=args.dropout,
                nlayer=args.nlayer)


model = utils.load_model(args,model,name='best')

model = model.to(device)

#prepare aligner for the explainer
if args.aligner == 'emb':
    aligner = aligners.EmbAligner(need_grad=args.align_with_grad, align_weight=args.align_weight)
elif args.aligner == 'anchor':
    aligner = aligners.AnchorAligner(need_grad=args.align_with_grad, align_weight=args.align_weight)
    aligner.set_anchors(dataset,model, args)
elif args.aligner == 'both':
    aligner = aligners.BothAligner(need_grad=args.align_with_grad, align_weight=args.align_weight, aligner_combine_weight=args.aligner_combine_weight)
    aligner.set_anchors(dataset,model, args)
elif args.aligner == 'MixGaus':
    aligner = aligners.MixGaussAligner(need_grad=args.align_with_grad, align_weight=args.align_weight)
    aligner.set_anchors(dataset,model, args)
elif args.aligner == 'MI':
    layer_id=1
    aligner = aligners.MIAligner(need_grad=args.align_with_grad, align_weight=args.align_weight,layer_id=layer_id, args=args,)
    mi_est_path = './checkpoint/{}/{}/MI_layer{}.pth'.format(args.dataset,args.model,layer_id)
    
    if os.path.exists(mi_est_path):
        # load the mi estimator
        loaded_content = torch.load(mi_est_path, map_location=lambda storage, loc: storage)
        aligner.mi_head.load_state_dict(loaded_content['mi'])
    else:
        #train the mi estimator
        aligner.train(dataset,model, args)
        saved_content = {}
        saved_content['mi'] = aligner.mi_head.state_dict()
        torch.save(saved_content, mi_est_path)


###explain
kwargs = {}
kwargs['edge_size']=args.edge_size
kwargs['edge_ent']=args.edge_ent
kwargs['nlayer']=args.nlayer
kwargs['directional']=args.directional
if args.explainer.lower() == 'gnnexplainer':
    explainer = GNNExplainer(model, type=args.datatype, epochs=args.epochs, lr=args.lr, loss=args.expl_loss, align_emb=args.align_emb, aligner=aligner, **kwargs)
elif args.explainer.lower() == 'pgexplainer':
    explainer = PGExplainer(model, type=args.datatype,  epochs=args.epochs, lr=args.lr,loss=args.expl_loss, align_emb=args.align_emb, aligner=aligner, **kwargs)
elif args.explainer.lower() == 'pgexplainer2':
    explainer = PGExplainer2(model, type=args.datatype, epochs=args.epochs, lr=args.lr, loss=args.expl_loss,align_emb=args.align_emb, aligner=aligner,  **kwargs)

explainer.prepare(dataset, args)

if args.datatype == 'node':
    data = dataset[0]
    data = data.to(next(model.parameters()).device)
    nodes = dataset[0].expl_mask.nonzero().squeeze().tolist()
    #if len(nodes) > 50:
    #    nodes = nodes[:50]

    AUC_meters = utils.meters()
    Tgt_acc = utils.meters()#prediction accuracy on the target node
    result_list = []
    for node_idx in nodes:
        #obtain ground-truth explanation for path-structured explanation on target node
        if 'edge_expl_path' in data.keys:
            data.edge_label = utils.path2mask(node_idx, data.edge_expl_path[node_idx], data.edge_index)
        node_feat_mask, edge_mask, edge_index, center_node, kwargs = explainer.explain_node(node_idx, data.x, data.edge_index, y=data.y, edge_label=data.edge_label,feat=data.x)
        
        #collect explanation results
        expl_dict={'center_node': center_node, 'x':kwargs.get('feat'), 'edge_index':edge_index, 'node_feat_mask':node_feat_mask, 'edge_mask':edge_mask, 'y_gt':kwargs.get('y')}
        result_list.append(expl_dict)

        #calculate AUC-ROC score on obtained explanations
        if len(kwargs.get('edge_label').cpu().unique()) > 1:# do not consider all-zero or all-one instances
            auc_score = roc_auc_score(kwargs.get('edge_label').detach().cpu(), edge_mask.detach().cpu())
            if args.log:
                writer.add_scalar('auroc of edge explanation', auc_score, node_idx)
            AUC_meters.update(auc_score,weight=1.0)
        #log classification accuracy information
        Tgt_acc.update(kwargs.get('pred_y')[center_node]==kwargs.get('y')[center_node], weight=1)

        #draw expl graph
        show_edge_num = kwargs['edge_label'].nonzero().shape[0]
        show_edge_num = int(show_edge_num*1.5) if show_edge_num*1.5 < edge_index.shape[1] else show_edge_num
        threshold = edge_mask.topk(k=show_edge_num)[0][-1].item()
        fig, G = explainer.visualize_subgraph(center_node, edge_index, edge_mask, y=kwargs['pred_y'],threshold=threshold)
        fig_gt, _ = explainer.visualize_subgraph(center_node, edge_index, kwargs['edge_label'], y=kwargs['y'])
        if args.log:
            writer.add_figure('G_exp',fig, node_idx)
            writer.add_figure('G_gt',fig_gt, node_idx)

        #draw edge weight distribution
        fig = plot_dist1D(edge_mask.detach().cpu().numpy(), label=kwargs.get('edge_label').cpu().int().numpy())
        if args.log:
            writer.add_figure('edge_mask',fig, node_idx)
        plt.close('all')
elif args.datatype == 'graph':
    #if len(graph_list) > 50:
    #    graph_list = graph_list[:50]

    AUC_meters = utils.meters()
    Tgt_acc = utils.meters()#prediction accuracy on the target node
    result_list = []
    for graph_id, data in enumerate(graph_list):
        data = data.to(next(model.parameters()).device)
        batch = data.x.new(data.x.shape[0]).long().fill_(0)

        #obtain ground-truth explanation for path-structured explanation on target node
        node_feat_mask, edge_mask, edge_index, center_node, kwargs = explainer.explain_graph(data.x, data.edge_index, y=data.y, edge_label=data.edge_label, batch=batch)
        
        #collect explanation results
        expl_dict={'center_node': -1, 'x':data.x, 'edge_index':edge_index, 'node_feat_mask':node_feat_mask, 'edge_mask':edge_mask, 'y_gt':kwargs.get('y')}
        result_list.append(expl_dict)

        #calculate AUC-ROC score on obtained explanations
        if len(kwargs.get('edge_label').cpu().unique()) > 1:# do not consider all-zero or all-one instances
            auc_score = roc_auc_score(kwargs.get('edge_label').detach().cpu(), edge_mask.detach().cpu())
            if args.log:
                writer.add_scalar('auroc of edge explanation', auc_score, graph_id)
            AUC_meters.update(auc_score,weight=1.0)
        #log classification accuracy information
        Tgt_acc.update(kwargs.get('pred_y')[0]==kwargs.get('y')[0], weight=1)

        #draw expl graph
        show_edge_num = kwargs['edge_label'].nonzero().shape[0]
        show_edge_num = int(show_edge_num*1.5) if show_edge_num*1.5 < edge_index.shape[1] else show_edge_num
        threshold = edge_mask.topk(k=show_edge_num)[0][-1].item()
        fig, G = explainer.visualize_subgraph(-1, edge_index.cpu(), edge_mask.cpu(), y=kwargs['pred_y'].cpu(),threshold=threshold)
        fig_gt, _ = explainer.visualize_subgraph(-1, edge_index.cpu(), kwargs['edge_label'].cpu(), y=kwargs['y'].cpu())
        if args.log:
            writer.add_figure('G_exp',fig, graph_id)
            writer.add_figure('G_gt',fig_gt, graph_id)

        #draw edge weight distribution
        fig = plot_dist1D(edge_mask.detach().cpu().numpy(), label=kwargs.get('edge_label').cpu().int().numpy())
        if args.log:
            writer.add_figure('edge_mask',fig, graph_id)
        plt.close('all')

# log fidelity analysis =
Pos_fidel_list, Neg_fidel_list, Pos_fidel_std, Neg_fidel_std = explainer.fidelity_analysis(result_list,k=20, with_std=True)
fig_pos = plot_chart([Pos_fidel_list],[args.explainer], std_lists=[Pos_fidel_std], x_start=1,y_name='Agreement Rate', val_min=0, val_max=1)
fig_neg = plot_chart([Neg_fidel_list],[args.explainer], std_lists=[Neg_fidel_std], x_start=1, y_name='Agreement Rate', val_min=0, val_max=1)
if args.log:
    writer.add_figure('Pos_fidelity',fig_pos, 1)
    writer.add_figure('Neg_fidelity',fig_neg, 1)

#save fidelity analysis result
if True:
    np.save(tb_path+'/pos_avg', np.array(Pos_fidel_list))
    np.save(tb_path+'/neg_avg', np.array(Neg_fidel_list))
    np.save(tb_path+'/pos_std', np.array(Pos_fidel_std))
    np.save(tb_path+'/neg_std', np.array(Neg_fidel_std))



# get embedding distance plot and visualize
'''
emb_dists, agreements = explainer.embed_distance_analysis(result_list)
fig = plot_dist1D(emb_dists.detach().cpu().numpy(), label=agreements.cpu().int().numpy())
if args.log:
    writer.add_figure('edge_mask',fig, node_idx)
'''

# log explain loss
print('Edge explanation auc score: {}'.format(AUC_meters.avg()))
if args.log:
    writer.add_scalar('Edge explanation avg-AUC', AUC_meters.avg(), 1)
    writer.add_scalar('Tgt classification accuracy', Tgt_acc.avg(), 1)


if args.log:
    writer.close()

print("Optimization Finished!")
