
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

import sys
sys.path.append("..")

class Benchmark(object):
    '''
    Parent class define basic parameters and functions of benchmarks. Set explanation methods and hyperparameters for training model.
    '''
    METHODS = [
    'pgmexplainer',
    'gnnexplainer',
    'sa',
    'random',
    'ig'           ]
    LR = 0.003
    EPOCHS = 400
    WEIGHT_DECAY = 0

    def __init__(self, sample_count):
        arguments = {
            'explanation_methods':self.METHODS,
            'sample_count': sample_count,
            'learn_rate': self.LR,
            'train_epochs': self.EPOCHS,
            'weight_decay': self.WEIGHT_DECAY
        }
        self.sample_count = sample_count

        mlflow.log_params(arguments)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def create_dataset(self):
        raise NotImplementedError

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name):
        raise NotImplementedError

    def train(self, model, optimizer, train_loader):
        raise NotImplementedError

    def test(self, model, loader):
        raise NotImplementedError

    def train_and_test(self, model, train_loader, test_loader):
        raise NotImplementedError

    def is_trained_model_valid(self, test_acc):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError