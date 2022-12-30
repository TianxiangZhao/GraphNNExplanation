import torch
import os
import argparse
import json

from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F
import ipdb

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--sparse', action='store_true', default=False,
                    help='whether use sparse adj matrix')
    parser.add_argument('--seed', type=int, default=4)
    
    parser.add_argument('--datatype', type=str, default='node', choices=['node', 'graph'])
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'gcls', 'expl'])#cls: node classification; gcls: graph classification; expl: explanation
    parser.add_argument('--dataset', type=str, default='BA_shapes') #choices=['BA_shapes','infected','Tree_cycle','Tree_grid','LoadBA_shapes', 'LoadTree_cycle','LoadTree_grid','mutag', 'SpuriousMotif_{}'.format(mix_ratio), 'SST2','SST5','Twitter']
    
    parser.add_argument('--nlayer', type=int, default=2)#intermediate feature dimension
    parser.add_argument('--nhid', type=int, default=20)#intermediate feature dimension
    parser.add_argument('--nclass', type=int, default=5)#number of labels
    parser.add_argument('--nfeat', type=int, default=64) # input feature dimension
    parser.add_argument('--epochs', type=int, default=510,
                help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1,
                help='Number of batches inside an epoch.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')

    parser.add_argument('--load', type=int, default=None) #load from pretrained model under the same setting, indicate load epoch
    parser.add_argument('--save', action='store_true', default=False,help='whether save checkpoints')
    parser.add_argument('--log', action='store_true', default=False,
                    help='whether creat tensorboard logs')
    parser.add_argument('--load_model', type=str, default=None) #To indicate pre-train in other folders. Like "./checkpoint/SpuriousMotif_0.3/best".
    

    parser.add_argument('--model', type=str, default='gcn', 
        choices=['sage','gcn','GAT','sage2','MLP','RGCN','HAN', 'DISGAT', 'GIN', 'FactorGCN', 'Mixhop', 'H2GCN'])
    parser.add_argument('--shared_encoder', action='store_true', default=False,help='False: train one end-to-end model; True: for multi-task, train a shared encoder with task-wise heads')
    
    parser.add_argument('--load_config', action='store_true', default=False, help='whether load training configurations')

    #explainer choices
    parser.add_argument('--explainer', type=str, default='gnnexplainer', choices=['gnnexplainer', 'pgexplainer','pgexplainer2' ])
    parser.add_argument('--directional', action='store_true', default=False, help='whether taking graph as directional or not in explanation')
    parser.add_argument('--edge_size', type=float, default=0.05, help='control edge mask sparsity')
    parser.add_argument('--edge_ent', type=float, default=1.0, help='control edge entropy weight')
    parser.add_argument('--expl_loss', type=str, default='Tgt', choices=['Tgt', 'Entropy','Dif' ])#
    parser.add_argument('--aligner', type=str, default='emb', choices=['emb', 'anchor','both', 'MixGaus', 'MI'])#
    parser.add_argument('--aligner_combine_weight', type=float, default=1.0)#
    parser.add_argument('--align_emb', action='store_true', default=False,  help='whether aligning embeddings in obtaining explanation')
    parser.add_argument('--align_with_grad', action='store_true', default=False,  help='whether aligning embeddings in obtaining explanation with gradient-based weighting')
    parser.add_argument('--align_weight', type=float, default=1.0)


    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()

    if args.load_config:
        config_path='./configs/{}/{}/{}'.format(args.task,args.dataset,args.model)
        with open(config_path) as f:
            args.__dict__ = json.load(f)

    return args

def save_args(args):
    config_path='./configs/{}/{}/'.format(args.task,args.dataset)

    if not os.path.exists(config_path):
        os.makedirs(config_path)

    with open(config_path+args.model,'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    return



def save_model(args, model, epoch=None, name='model'):
    saved_content = {}
    saved_content[name] = model.state_dict()

    path = './checkpoint/{}/{}'.format(args.dataset, args.model)
    if not os.path.exists(path):
        os.makedirs(path)

    #torch.save(saved_content, 'checkpoint/{}/{}_epoch{}_edge{}_{}.pth'.format(args.dataset,args.model,epoch, args.used_edge, args.method))
    if epoch is not None:
        torch.save(saved_content, os.path.join(path,'{}_{}.pth'.format(name, epoch)))
        print("successfully saved: {}".format(epoch))
    else:
        torch.save(saved_content, os.path.join(path,'{}.pth'.format(name)))
        print("successfully saved: {}".format(name))

    return

def load_model(args, model, name='model_500'):
    
    loaded_content = torch.load('./checkpoint/{}/{}/{}.pth'.format(args.dataset, args.model,name), map_location=lambda storage, loc: storage)

    model.load_state_dict(loaded_content['best'])

    print("successfully loaded: {}.pth".format(name))

    return model

def load_specific_model(model, name='./checkpoint/mutag/gcn/model_500.pth'):
    
    loaded_content = torch.load(name, map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded_content['best'])

    print("successfully loaded: {}".format(name))

    return model

def path2mask(head_node, path, edge_index):
    #change path to edge mask
    edge_mask = edge_index.new(edge_index.shape[1]).fill_(0)
    
    cur_ind = 0
    while path[cur_ind+1] != -1:
        src_node = path[cur_ind+1]
        tgt_node = path[cur_ind]

        edge_ind = torch.mul(edge_index[0]==src_node, edge_index[1]==tgt_node).nonzero().squeeze().item()
        edge_mask[edge_ind] = 1
        cur_ind += 1

    return edge_mask.to(int)

def accuracy(logits, labels):
    preds = logits.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(logits, labels, pre='valid'):
    pre_num = 0
    #print class-wise performance
    
    for i in range(labels.max()+1):
        index_pos = labels==i
        cur_tpr = accuracy(logits[index_pos], labels[index_pos])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_neg = labels != i
        labels_neg = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(logits[index_neg,:], labels_neg[index_neg])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))
    

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return

def Roc_F(logits, labels, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(logits[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(logits[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return auc_score, macro_F

class meters:
    '''
    collects the results at each inference batch, and return the result in total
    param orders: the order in updating values
    '''
    def __init__(self, orders=1):
        self.avg_value = 0
        self.tot_weight = 0
        self.orders = orders
        
    def update(self, value, weight=1.0):
        value = float(value)

        if self.orders == 1:
            update_step = self.tot_weight/(self.tot_weight+weight)
            self.avg_value = self.avg_value*update_step + value*(1-update_step)
            self.tot_weight += weight
        

    def avg(self):

        return self.avg_value