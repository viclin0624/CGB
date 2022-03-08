import json
import os
import tempfile
import time

import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
import sys
import dgl
import numpy as np
import pandas as pd
from tqdm import tqdm as tq
import mlflow
from collections import defaultdict
sys.path.append('..')
from model.models_dgl import FixedNet2

from build_graph import BA4labelDataset,build_graph
from benchmark_dgl import Benchmark
from method.explain_methods_dgl import explain_random, explain_ig, explain_sa, explain_gnnexplainer, explain_pgmexplainer

def generate_single_sample(label, perturb_type, nodes_num = 25, m = 1, perturb_dic = {}, 
    seed = None, no_attach_init_nodes=False):
    '''
    This function is for test model return a networkx instance.
    '''
    basis_type = "ba"
    which_type = label
    if which_type == 0:
        if perturb_type == 0:
            G, role_id, plug_id = build_graph(nodes_num, basis_type, [], start = 0, m = m, seed = seed, no_attach_init_nodes=no_attach_init_nodes)
        else:
            G, role_id, plug_id = build_graph(nodes_num - perturb_type, basis_type, [[perturb_dic[perturb_type]]], start = 0, m = m, seed = seed, no_attach_init_nodes=no_attach_init_nodes)
    else:
        list_shapes = [["house"]] * (which_type - 1) + [["five_cycle"]] * (3 - which_type)
        if perturb_type != 0:
            list_shapes = list_shapes + [[perturb_dic[perturb_type]]]
        G, role_id, plug_id = build_graph(nodes_num-10-perturb_type, basis_type, list_shapes, start = 0, m = m, seed = seed, no_attach_init_nodes=no_attach_init_nodes)
    return G

def test_model_fixed(model_test, graphs_num = 1000, m = 5, nodes_num = 25, perturb_dic = {}, device = torch.device('cuda'), no_attach_init_nodes = False):
    model_test.eval()
    data = BA4labelDataset(graphs_num=graphs_num, m = m, nodes_num=nodes_num, perturb_dic = perturb_dic, no_attach_init_nodes = no_attach_init_nodes)
    testdataloader = dgl.dataloading.GraphDataLoader(data, batch_size = 16, shuffle = True)
    correct = 0
    total = 0
    for g, labels in testdataloader:
        g = g.to(device)
        output = model_test(g, g.ndata['x'])
        pred = output.max(dim=1)[1]
        eq_pred = pred.eq(labels)
        correct += eq_pred.sum().item()
        if eq_pred.sum().item() != eq_pred.shape[0]:
            return g,eq_pred,labels
        total += len(labels.to(device))
    test_acc = correct/total
    return test_acc

class BA4label(Benchmark):
    NUM_TRAIN_GRAPHS = 1000
    NUM_NODES = 50
    TEST_RATIO = 0.4
    LR = 0.005
    M = 6
    NO_ATTACH_INIT_NODES = True

    @staticmethod
    def get_accuracy(g, correct_ids, edge_mask):
        '''
        edge_index: 2 elements tuple, u and v
        '''
        if correct_ids == []:
            if np.max(edge_mask)==0 or np.all(np.mean(edge_mask) == edge_mask):
                return 1
            else:
                return 0
        else:
            correct_edges = set()
            for i in range(g.num_edges()):
                u = g.edges()[0][i].item()
                v = g.edges()[1][i].item()
                if u in correct_ids or v in correct_ids: #每个双向边都计入，是否要将边缘的边改为只计入一遍？
                    correct_edges.add((u,v))
                    correct_edges.add((v,u))
            #按照单向边来计算
            correct_count = 0
            for x in np.argsort(-edge_mask)[:len(correct_edges)]:
                u = g.edges()[0][x].item()
                v = g.edges()[1][x].item()
                if (u, v) in correct_edges:
                    correct_count += 1
            #双向边只要对一条就算对 注意 此时正确边要包含每个双向边
            '''
            mask_edges = set()
            for x in np.argsort(-edge_mask)[:len(correct_edges)]:
                u = g.edges[0][x].item()
                v = g.edges[1][x].item()
                if ( (u, v) in correct_edges or (v, u) in correct_edges ) and (v, u) not in mask_edges:
                    mask_edges.add((v,u))
                    correct_count += 2
            '''
            return correct_count / len(correct_edges)

    def create_dataset(self, graphs_num, m = 6, nodes_num = 50):
        '''
        Return data
        '''
        data = BA4labelDataset(graphs_num=graphs_num, m = m, nodes_num=nodes_num, perturb_dic = {}, no_attach_init_nodes = True)
        print('created one')
        return data

    def is_trained_model_valid(self, test_acc):
        return test_acc == 1

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name):
        accs = []
        tested_graphs = 0
        for g, label in tq(test_dataset):
            g = g.to(self.device)
            tested_graphs += 1
            edge_mask = explain_function(model, 'graph', g, g.ndata['x'], label)
            if label == 0:
                correct_ids = []
            else:
                correct_ids = [x for x in range(len(g.nodes())-10,len(g.nodes()))]
            explain_acc = self.get_accuracy(g, correct_ids, edge_mask)
            accs.append(explain_acc)
            mlflow.log_metric('tested_graphs', tested_graphs)
        return accs

    def train(self, model, train_loader):
        #don't need to train
        loss_all = 0
        for g, label in train_loader:
            g = g.to(self.device)
            label = label.to(self.device)
            output = model(g, g.ndata['x'])
            loss = F.nll_loss(output.float(), label)
            loss_all += loss.item()
        return loss_all / len(train_loader)

    def test(self, model, loader):
        model.eval()

        correct = 0
        total = 0
        for g, label in loader:
            g = g.to(self.device)
            label = label.to(self.device)
            output = model(g, g.ndata['x'])
            pred = output.max(dim=1)[1]
            correct += pred.eq(label).sum().item()
            total += len(label)
        return correct / total

    def train_and_test(self, model, train_loader, test_loader):
        train_loss = self.train(model, train_loader)
        train_acc = self.test(model, train_loader)
        test_acc = self.test(model, test_loader)
        print('train_loss:',train_loss, 'train_acc:',train_acc, 'test_acc:', test_acc)
        return train_acc, test_acc

    def run(self):
        print(f"Using device {self.device}")
        benchmark_name = self.__class__.__name__
        all_explanations = defaultdict(list)
        all_runtimes = defaultdict(list)
        for experiment_i in tq(range(self.sample_count)):
            train_dataset = self.create_dataset(self.NUM_TRAIN_GRAPHS)
            test_dataset = self.create_dataset(int(self.NUM_TRAIN_GRAPHS * self.TEST_RATIO))

            train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size = 1, shuffle = True)
            test_dataloader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size = 1, shuffle = True)
            model = FixedNet2(1, 4, 2, False, 'GraphConvWL').to(self.device)
            model.set_paramerters()
            model.to(self.device)
            train_acc, test_acc = self.train_and_test(model, train_dataloader, test_dataloader)
            if not self.is_trained_model_valid(test_acc):
                print('Model accuracy was not valid, ignoring this experiment')
                continue
            model.eval()
            metrics = {
                'train_acc': train_acc,
                'test_acc': test_acc,
            }
            mlflow.log_metrics(metrics, step=experiment_i)

            for explain_name in self.METHODS:
                explain_function = eval('explain_' + explain_name)
                duration_samples = []

                def time_wrapper(*args, **kwargs):
                    start_time = time.time()
                    result = explain_function(*args, **kwargs)
                    end_time = time.time()
                    duration_seconds = end_time - start_time
                    duration_samples.append(duration_seconds)
                    return result

                time_wrapper.explain_function = explain_function
                accs = self.evaluate_explanation(time_wrapper, model, test_dataset, explain_name)
                print(f'Benchmark:{benchmark_name} Run #{experiment_i + 1}, Explain Method: {explain_name}, Accuracy: {np.mean(accs)}')
                all_explanations[explain_name].append(list(accs))
                all_runtimes[explain_name].extend(duration_samples)
                metrics = {
                    f'explain_{explain_name}_acc': np.mean(accs),
                    f'time_{explain_name}_s_avg': np.mean(duration_samples),
                }
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, 'accuracies.json')
                    json.dump(all_explanations, open(file_path, 'w'), indent=2)
                    mlflow.log_artifact(file_path)
                mlflow.log_metrics(metrics, step=experiment_i)
            print(f'Benchmark:{benchmark_name} Run #{experiment_i + 1} finished. Average Explanation Accuracies for each method:')
            accuracies_summary = {}
            for name, run_accs in all_explanations.items():
                run_accs = [np.mean(single_run_acc) for single_run_acc in run_accs]
                accuracies_summary[name] = {'avg': np.mean(run_accs), 'std': np.std(run_accs), 'count': len(run_accs)}
                print(f'{name} : avg:{np.mean(run_accs)} std:{np.std(run_accs)}')
            runtime_summary = {}
            for name, runtimes in all_runtimes.items():
                runtime_summary[name] = {'avg': np.mean(runtimes), 'std': np.std(runtimes)}
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, 'summary.json')
                summary = {'accuracies': accuracies_summary, 'runtime': runtime_summary}
                json.dump(summary, open(file_path, 'w'), indent=2)
                mlflow.log_artifact(file_path)

def test_model_acc():
    model_fixed = FixedNet(1, 4, 2, False, 'GraphConvWL')
    model_fixed.set_paramerters()
    device = torch.device('cuda')
    model_fixed = model_fixed.to(device)
    model_fixed.unuse_report()
    G = generate_single_sample(3, 0, nodes_num = 25, m = 6, perturb_dic = {4:'square_diagonal'}, seed = 0, no_attach_init_nodes = True)
    pos = nx.spring_layout(G, seed = 0)
    nx.draw(G, pos, with_labels = True, font_color = 'white')
    g = dgl.from_networkx(G)
    g = g.to(device)
    output = model_fixed(g, torch.ones((25,1)).to(device))
    print(output)
    test_acc = test_model_fixed(model_fixed, graphs_num = 1000, m = 6, nodes_num = 50, no_attach_init_nodes = True)
    print(test_acc)

def test_model_output_distribution(graph_class,graph_num):
    model_fixed = FixedNet(1, 4, 2, False, 'GraphConvWL')
    model_fixed.set_paramerters()
    device = torch.device('cuda')
    model_fixed = model_fixed.to(device)
    model_fixed.unuse_report()
    result = []
    for i in range(graph_num):
        G = generate_single_sample(graph_class, 0, nodes_num = 50, m = 6, perturb_dic = {}, no_attach_init_nodes = True)
        G = dgl.from_networkx(G)
        cut_index = np.random.choice(list(range(G.num_edges())))
        G = dgl.remove_edges(G, cut_index)
        G = G.to(device)
        result.append(model_fixed(G, torch.ones((50,1)).to(device)))
    print(result)
    return result


if __name__ == '__main__':
    A = BA4label(1,1,True,'GraphConvWL')
    A.run()
    #test_model_output_distribution(3,20)