import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from tqdm import tqdm
import numpy as np
import ipdb

import utils
import models.models


def fuse_feature(feature_list, fuse='last'): 
    '''fuse features from each layer in the encoder
    Args:
        feature_list:
        fuse:

    '''
    if fuse == 'last':
        fused_feat = feature_list[-1]
    elif fuse == 'avg':
        fused_feat = torch.mean(torch.stack(feature_list))
    elif fuse == 'concat':
        fused_feat = torch.cat(feature_list, dim=-1)

    return fused_feat

def cal_feat_dim(args): 
    '''get dimension of obtained embedding feature
    Args: 
        args:
    '''

    emb_dim = args.nhid
    if args.fuse == 'concat':
        emb_dim = emb_dim * args.enc_layer

    return emb_dim


class Trainer(object):
    def __init__(self, args, model, weight):#
        self.args = args

        self.loss_weight = weight
        self.models = []
        self.models.append(model)

        self.models_opt = []
        for model in self.models:
            self.models_opt.append(optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay))

    def train_step(self, data, pre_adj):#pre_adj corresponds to adj used for generating ssl signal
        raise NotImplementedError('train not implemented for base class')

    def inference(self, data):
        raise NotImplementedError('infer not implemented for base class')

        
class ClsTrainer(Trainer):
    '''for node classification
    
    '''
    def __init__(self, args, model, weight=1.0):
        super().__init__(args, model, weight)        

        self.args = args
        
        if args.shared_encoder:#provided model is the shared feature encoder, and trainer-specific classifier is required for each task
            self.in_dim = cal_feat_dim(args)
            self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=labels.max().item() + 1, layers=args.cls_layer)
            if args.cuda:
                self.classifier.cuda()
            self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
            self.models.append(self.classifier)
            self.models_opt.append(self.classifier_opt)


    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        acc_train = utils.accuracy(output[data.train_mask], data.y[data.train_mask])

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for opt in self.models_opt:
            opt.step()
        
        #ipdb.set_trace()

        loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        acc_val = utils.accuracy(output[data.val_mask], data.y[data.val_mask])
        #utils.print_class_acc(output[data.val_mask], data.y[data.val_mask])
        roc_val, macroF_val = utils.Roc_F(output[data.val_mask], data.y[data.val_mask])

        
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()))
        
        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                     'loss_val': loss_val.item(), 'acc_val': acc_val.item(), 'roc_val': roc_val, 'macroF_val': macroF_val }

        return log_info

    def test(self, data):
        for i, model in enumerate(self.models):
            model.eval()


        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        acc_test = utils.accuracy(output[data.test_mask], data.y[data.test_mask])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output[data.test_mask], data.y[data.test_mask], pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output[data.test_mask], data.y[data.test_mask], pre='test')
        
        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info

class GClsTrainer(Trainer):
    '''for node classification
    
    '''
    def __init__(self, args, model, weight=1.0):
        super().__init__(args, model, weight)        

        self.args = args
        
        if args.shared_encoder:#provided model is the shared feature encoder, and trainer-specific classifier is required for each task
            self.in_dim = cal_feat_dim(args)
            self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=labels.max().item() + 1, layers=args.cls_layer)
            if args.cuda:
                self.classifier.cuda()
            self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
            self.models.append(self.classifier)
            self.models_opt.append(self.classifier_opt)


    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        if self.args.shared_encoder:#not tested yet
            ipdb.set_trace()
            output = self.get_em(data)#need to include batch informaiton
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_train = F.nll_loss(output, data.y)
        acc_train = utils.accuracy(output, data.y)

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for opt in self.models_opt:
            opt.step()
        
        #ipdb.set_trace()

        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)

        '''
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),)
        '''
        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                    }

        return log_info

    def test(self, data):
        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder: #not tested yet
            ipdb.set_trace()
            output = self.get_em(data)#need to revise for graphs
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_test = F.nll_loss(output, data.y)
        acc_test = utils.accuracy(output, data.y)

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output, data.y, pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output, data.y, pre='test')
        
        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info