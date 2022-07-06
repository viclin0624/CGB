
import json
import os
import tempfile
import time
from collections import defaultdict
import random

import mlflow
import torch.nn.functional as F
from tqdm import tqdm as tq

import networkx as nx
import numpy as np
import torch

#from explain_methods import *
#from models import Net1
import sys
sys.path.append("..")

class Benchmark(object):
    '''
    三个Benchmark的父类
    
    需要子类继承：create_dataset和evaluate_explanation这两个方法
    '''
    NUM_GRAPHS = 2
    TEST_RATIO = 0.5
    PGMEXPLAINER_SUBSAMPLE_PER_GRAPH = 20
    METHODS = [
    'pgmexplainer',
    'gnnexplainer',
    'sa',
    'random',
    'ig'           ]
    LR = 0.0003
    EPOCHS = 400
    WEIGHT_DECAY = 0

    def __init__(self, sample_count, num_layers, concat_features, conv_type):
        arguments = {
            'sample_count': sample_count,
            'num_layers': num_layers,
            'concat_features': concat_features,
            'conv_type': conv_type,
            'num_graphs': self.NUM_GRAPHS,
            'test_ratio': self.TEST_RATIO,
        }
        self.sample_count = sample_count
        self.num_layers = num_layers
        self.concat_features = concat_features
        self.conv_type = conv_type
        mlflow.log_params(arguments)
        mlflow.log_param('PGMEXPLAINER_SUBSAMPLE_PER_GRAPH', self.PGMEXPLAINER_SUBSAMPLE_PER_GRAPH)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def create_dataset(self):
        raise NotImplementedError

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name):
        raise NotImplementedError

    def subsample_nodes(self, explain_function, nodes):
        #if explain_function.explain_function != explain_pgmexplainer:
        #    return nodes
        return random.sample(nodes, self.PGMEXPLAINER_SUBSAMPLE_PER_GRAPH)

    @staticmethod
    def aggregate_directions(edge_mask, edge_index):
        edge_values = defaultdict(float)
        for x in range(len(edge_mask)):
            u, v = edge_index[:, x]
            u, v = u.item(), v.item()
            if u > v:
                u, v = v, u
            edge_values[(u, v)] += edge_mask[x]
        return edge_values

    def train(self, model, optimizer, train_loader):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data, data.ndata['x'])
            loss = F.nll_loss(output, data.ndata['y'])
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
        return loss_all / len(train_loader)

    def test(self, model, loader):
        model.eval()

        correct = 0
        total = 0
        for data in loader:
            data = data.to(self.device)
            output = model(data, data.ndata['x'])
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.ndata['y']).sum().item()
            total += len(data.ndata['y'])
        return correct / total

    def train_and_test(self, model, train_loader, test_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        mlflow.log_param('weight_decay', self.WEIGHT_DECAY)
        mlflow.log_param('lr', self.LR)
        mlflow.log_param('epochs', self.EPOCHS)
        pbar = tq(range(self.EPOCHS))
        for epoch in pbar:
            train_loss = self.train(model, optimizer, train_loader)
            train_acc = self.test(model, train_loader)
            test_acc = self.test(model, test_loader)
            pbar.set_postfix(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)
        return train_acc, test_acc

    def is_trained_model_valid(self, test_acc):
        return True

    def run(self):
        raise NotImplementedError