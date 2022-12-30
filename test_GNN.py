import torch
import numpy as np
import random
import os
import datasets as datasets
from torch_geometric.loader import DataLoader,RandomNodeSampler
import models.models as models
import utils
import trainers.trainer as trainer
import math
import torch_geometric.datasets as tg_dataset

from tensorboardX import SummaryWriter
import ipdb


###configure arguments
args = utils.get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.log:
    tb_path = './tensorboard/{}/{}/model{}/'.format(args.dataset,args.task, args.model,)
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
        graph_num = len(dataset)
        dataloader = DataLoader(dataset[:int(graph_num*0.5)], batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(dataset[int(graph_num*0.5):], batch_size=len(dataset[int(graph_num*0.5):]), shuffle=False)
    if args.dataset == 'Twitter':
        dataset = datasets.SentiGraphDataset(root='./datasets/SentiGraph', name='Graph-Twitter')
        args.nclass = 3
        graph_num = len(dataset)
        dataloader = DataLoader(dataset[:int(graph_num*0.5)], batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(dataset[int(graph_num*0.5):], batch_size=len(dataset[int(graph_num*0.5):]), shuffle=False)
    if args.dataset == 'SST2':
        dataset = datasets.SentiGraphDataset(root='./datasets/SentiGraph', name='Graph-SST2')
        args.nclass = 2
        graph_num = len(dataset)
        dataloader = DataLoader(dataset[:int(graph_num*0.5)], batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(dataset[int(graph_num*0.5):], batch_size=len(dataset[int(graph_num*0.5):]), shuffle=False)
    if args.dataset == 'SST5':
        dataset = datasets.SentiGraphDataset(root='./datasets/SentiGraph', name='Graph-SST5')
        args.nclass = 5
        graph_num = len(dataset)
        dataloader = DataLoader(dataset[:int(graph_num*0.5)], batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(dataset[int(graph_num*0.5):], batch_size=len(dataset[int(graph_num*0.5):]), shuffle=False)
    elif args.dataset.split('_')[0] == 'SpuriousMotif':
        name, mix_ratio = args.dataset.split('_')
        mix_ratio = float(mix_ratio)
        dataset = datasets.SyncGraphs(name, mix_ratio=mix_ratio)
        dataset = dataset.shuffle()
        args.nclass = 3
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        #testloader = dataloader
        testloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    args.nfeat = dataset[0].x.shape[-1]

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


if args.load_model is not None:
    model = utils.load_specific_model(model,name='{}.pth'.format(args.load_model))
else:
    model = utils.load_model(args,model,name='best')

model = model.to(device)

###initialize trainer
trainers=[]
Trainer_dict={'cls': trainer.ClsTrainer, 'gcls':trainer.GClsTrainer}

DOWNtrainer = Trainer_dict[args.task](args, model)
trainers.append(DOWNtrainer)

for trainer in trainers:
    #test all data inside dataset
    log_test = {}
    for batch, data in enumerate(testloader):
        log_info = trainer.test(data.to(device))
        print('test result at batch {}: {}'.format(batch, log_info))
        for key in log_info:
            if key not in log_test.keys():
                log_test[key] = utils.meters(orders=1)
                log_test[key].update(log_info[key], data.num_graphs)                
    for key in log_info:
        if args.log:
            writer.add_scalar(key, log_test[key].avg(), 0)
        print('test result summary on {}: {}'.format(key, log_test[key].avg()))

if args.log:
    writer.close()

print("Optimization Finished!")
