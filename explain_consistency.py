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
import explainers.EmbAligner as aligners

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
    tb_path = './tensorboard_journal/{}/{}_{}_stability/model{}/edgesize{}ent{}epoch{}exploss{}directional{}align{}aligner{}grad{}weight{}'.format(args.dataset,args.task, args.explainer, args.model,args.edge_size,args.edge_ent, args.epochs, args.expl_loss,args.directional, args.align_emb,args.aligner, args.align_with_grad, args.align_weight)
    
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

#aligners
if args.aligner == 'emb':
    aligner = aligners.EmbAligner(need_grad=args.align_with_grad, align_weight=args.align_weight)
elif args.aligner == 'anchor':
    aligner = aligners.AnchorAligner(need_grad=args.align_with_grad, align_weight=args.align_weight)
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
explainers = []
expl_num = 5
if args.explainer.lower() == 'gnnexplainer':
    for i in range(expl_num):
        explainer = GNNExplainer(model, type=args.datatype,  epochs=args.epochs, lr=args.lr, loss=args.expl_loss, align_emb=args.align_emb, aligner=aligner, **kwargs)
        explainers.append(explainer)
elif args.explainer.lower() == 'pgexplainer':
    for i in range(expl_num):
        explainer = PGExplainer(model, type=args.datatype,  epochs=args.epochs, lr=args.lr,loss=args.expl_loss, align_emb=args.align_emb, aligner=aligner, **kwargs)
        explainers.append(explainer)
elif args.explainer.lower() == 'pgexplainer2':
    for i in range(expl_num):
        explainer = PGExplainer2(model, type=args.datatype,  epochs=args.epochs, lr=args.lr, loss=args.expl_loss, align_emb=args.align_emb, aligner=aligner, **kwargs)
        explainers.append(explainer)

for explainer in explainers:
    explainer.prepare(dataset, args)

if args.datatype =='node':
    data = dataset[0]
    data = data.to(next(model.parameters()).device)
    nodes = dataset[0].expl_mask.nonzero().squeeze().tolist()
    #if len(nodes) > 50:
    #    nodes = nodes[:50]
        
    AUC_meters = utils.meters()
    expl_lists = []
    embedding_lists = []

    for node_idx in nodes:
        expl_list = []
        embedding_list = []
        if 'edge_expl_path' in data.keys:
            data.edge_label = utils.path2mask(node_idx, data.edge_expl_path[node_idx], data.edge_index)
        for expl_ind, explainer in enumerate(explainers):
            node_feat_mask, edge_mask, edge_index, center_node, kwargs = explainer.explain_node(node_idx, data.x, data.edge_index, y=data.y, edge_label=data.edge_label,feat=data.x)
            #embedding = explainer.explain_node_emb(node_idx, data.x, data.edge_index, y=data.y, edge_label=data.edge_label,feat=data.x)
            
            #calculate AUC-ROC score on obtained explanations
            if len(kwargs.get('edge_label').cpu().unique()) > 1:
                auc_score = roc_auc_score(kwargs.get('edge_label').detach().cpu(), edge_mask.detach().cpu())
                AUC_meters.update(auc_score,weight=1.0)

            expl_list.append(edge_mask)
            #embedding_list.append(embedding)
        expl_lists.append(expl_list)
        #embedding_lists.append(embedding_list)
elif args.datatype =='graph':
    #if len(graph_list) > 50:
    #    graph_list = graph_list[:50]
        
    AUC_meters = utils.meters()
    expl_lists = []
    embedding_lists = []

    AUC_meters_sep = [utils.meters() for number in range(expl_num)]
    for graph_id,data in enumerate(graph_list):
        data = data.to(next(model.parameters()).device)
        data.batch = data.x.new(data.x.shape[0]).long().fill_(0)
        expl_list = []
        embedding_list = []

        for expl_ind, explainer in enumerate(explainers):
            node_feat_mask, edge_mask, edge_index, center_node, kwargs = explainer.explain_graph(data.x, data.edge_index, y=data.y, edge_label=data.edge_label,batch=data.batch)
            
            #calculate AUC-ROC score on obtained explanations
            if len(kwargs.get('edge_label').cpu().unique()) > 1:
                auc_score = roc_auc_score(kwargs.get('edge_label').detach().cpu(), edge_mask.detach().cpu())
                AUC_meters.update(auc_score,weight=1.0)
                AUC_meters_sep[expl_ind].update(auc_score,weight=1.0)

            expl_list.append(edge_mask)
            
            #embedding_list.append(embedding)
        expl_lists.append(expl_list)
        #embedding_lists.append(embedding_list)

# log explain loss
print('Edge explanation auc score: {}'.format(AUC_meters.avg()))
if args.log:
    writer.add_scalar('Edge explanation avg-AUC', AUC_meters.avg(), 1)
#for ind, auc_sep in enumerate(AUC_meters_sep):
#    print('{}s explainer auroc: {}'.format(ind, auc_sep.avg()))

#analysis consistency on obtained expl
dif_lists = []
dif_to_k = []
dis_to_k = []
dif_to_k_std = []
for Edge_k in range(20):
    dif_list, dif_k, dis_k = explainer.stability_analysis(expl_lists, Edge_k=Edge_k+1)
    dif_lists.append(dif_list)
    dif_to_k.append(dif_k)
    dis_to_k.append(dis_k)
    dif_to_k_std.append(np.std(np.array(dif_list)))



# get embedding distance plot and visualize


#plot stability
fig = plot_chart([np.array(dif_to_k)], name_list=['stability_to_k on edge'], x_start=1, std_lists=[np.array(dif_to_k_std)])
if args.log:
    writer.add_figure('stability of top-k in terms of SHD distance',fig, 1)

fig = plot_chart([np.array(dis_to_k)], name_list=['stability_to_k on distance'], x_start=1)
if args.log:
    writer.add_figure('stability of top-k on explanation weight distribution',fig, 1)

fig = plot_dist1D(np.array(dif_lists[3]), label=None)#plot instance-wise SHD difference distribution for top-3 edges
if args.log:
    writer.add_figure('stability dist on SHD distance w.r.t instances',fig, 1)

#if args.log:
#    writer.add_figure('emb distance sample-wise distribution',fig, 1)



if args.log:
    writer.close()

print("Optimization Finished!")
