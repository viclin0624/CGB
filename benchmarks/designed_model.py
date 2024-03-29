import json
import os
import tempfile
import time
from matplotlib.collections import Collection
from copy import deepcopy
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
from model.models_dgl import GCN_designed
from collections import Counter
from benchmarks.build_graph import CGBDataset,build_graph
from benchmarks.benchmark_dgl import Benchmark
from method.explain_methods_dgl import explain_random, explain_ig, explain_sa, explain_gnnexplainer, explain_pgmexplainer

class CGB(Benchmark):
    # number of graphs in dataset
    NUM_TRAIN_GRAPHS = 100
    NUM_TEST_GRAPHS = 100
    # dataset parameters
    NUM_NODES = 50
    M = 5
    # others
    ITERATION_NUM_OF_SUMMARY = 10 # num of running explanation methods times, if 1, only have original results
    NO_ATTACH_INIT_NODES = True # motifs should not be attached to non-initial nodes in BA graph

    @staticmethod
    def get_accuracy(g, correct_ids, edge_mask):
        '''
        Calculate accuracy from correct nodes id and edge mask.
        '''
        if correct_ids == []:
            if np.all(np.mean(edge_mask) == edge_mask):
                return 1
            else:
                return 0
        else:
            correct_edges = set()
            for i in range(g.num_edges()):
                u = g.edges()[0][i].item()
                v = g.edges()[1][i].item()
                if u in correct_ids and v in correct_ids: 
                    correct_edges.add((u,v))
                    correct_edges.add((v,u))
                elif v in correct_ids:
                    correct_edges.add((u,v))
                else:
                    continue
            #elements in ground truth and explanation set are directional edges
            correct_count = 0
            for x in np.argsort(-edge_mask)[:len(correct_edges)]:
                u = g.edges()[0][x].item()
                v = g.edges()[1][x].item()
                if (u, v) in correct_edges:
                    correct_count += 1
            return correct_count / len(correct_edges)

    def create_dataset(self, graphs_num, m = 6, nodes_num = 50):
        '''
        Return dataset
        '''
        data = CGBDataset(graphs_num=graphs_num, m = m, nodes_num=nodes_num, perturb_dic = {}, no_attach_init_nodes = True, include_bias_class=False)
        print('created one')
        return data

    def is_trained_model_valid(self, test_acc):
        return test_acc == 1

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name, iteration = 1):
        '''
        Evaluate a explanation method on test dataset.
        INPUT:
        -------------
        explain_function:    use which explanation method
        model           :    explained model
        test_dataset    :    test dataset
        explain_name    :    explanation method name
        iteration       :    how many times does the explanation method run, if not 1, will summarize explanation results by 4 methods
        OUTPUT:
        -------------
        accs            :    accuracies of the explanation method. If iteration is not 1, there will be accs_sum and some others, meaning use "sum" method to summarize explanation result
        '''
        accs = []
        tested_graphs = 0
        accs_sum = []
        accs_count13 = []
        accs_rank = []
        accs_count26 = []
        for g, label in tq(test_dataset):
            g = g.to(self.device)
            tested_graphs += 1
            edge_mask = explain_function(model, 'graph', g, g.ndata['x'], label)

            if iteration != 1: # iteration not 
                edge_mask_list = [deepcopy(edge_mask)]
                for _ in range(1,iteration):
                    new_edge_mask = explain_function(model, 'graph', g, g.ndata['x'], label)
                    edge_mask_list.append(new_edge_mask)
                
                for summarytype in ['sum','count13','count26','rank']:
                    edge_mask = deepcopy(edge_mask_list[0])
                    if summarytype == 'sum':
                        for i in range(1, iteration):
                            edge_mask += edge_mask_list[i]
                        sum_edge_mask = edge_mask
                    if summarytype == 'count26':#influence by threshold and not be appropriate to stable method
                        top_edges_index = list(np.argsort(-edge_mask)[:26])
                        for j in range(1, iteration):
                            edge_mask = edge_mask_list[j]
                            top_edges_index.extend(list(np.argsort(-edge_mask)[:26]))
                        edges_index_dic = Counter(top_edges_index)
                        edge_mask = np.zeros(g.num_edges())
                        for k,v in edges_index_dic.items():
                            edge_mask[k] = v 
                        count26_edge_mask = edge_mask
                    if summarytype == 'rank':
                        edges_index = np.argsort(-edge_mask)
                        edges_rank = {}
                        for i in range(len(edges_index)):
                            edges_rank[edges_index[i]] = i
                        for j in range(1, iteration):
                            edge_mask = edge_mask_list[j]
                            edges_index = np.argsort(-edge_mask)
                            for i in range(len(edges_index)):
                                edges_rank[edges_index[i]] += i
                        edge_mask = []
                        for i in range(g.num_edges()):
                            edge_mask.append(edges_rank[i])
                        edge_mask = -np.array(edge_mask)
                        rank_edge_mask = edge_mask
                    if summarytype == 'count13':#influence by threshold and not be appropriate to stable method
                        top_edges_index = list(np.argsort(-edge_mask)[:13])
                        for j in range(1, iteration):
                            edge_mask = edge_mask_list[j]
                            top_edges_index.extend(list(np.argsort(-edge_mask)[:13]))
                        edges_index_dic = Counter(top_edges_index)
                        edge_mask = np.zeros(g.num_edges())
                        for k,v in edges_index_dic.items():
                            edge_mask[k] = v 
                        count13_edge_mask = edge_mask
                if label == 0:
                    correct_ids = []
                else:
                    correct_ids = [x for x in range(len(g.nodes())-10,len(g.nodes()))]
                explain_acc = self.get_accuracy(g, correct_ids, sum_edge_mask)
                accs_sum.append(explain_acc)
                explain_acc = self.get_accuracy(g, correct_ids, count13_edge_mask)
                accs_count13.append(explain_acc)
                explain_acc = self.get_accuracy(g, correct_ids, rank_edge_mask)
                accs_rank.append(explain_acc)
                explain_acc = self.get_accuracy(g, correct_ids, count26_edge_mask)
                accs_count26.append(explain_acc)
                edge_mask = edge_mask_list[0]
                
            if label == 0:
                correct_ids = []
            else:
                correct_ids = [x for x in range(len(g.nodes())-10,len(g.nodes()))]

            explain_acc = self.get_accuracy(g, correct_ids, edge_mask)
            accs.append(explain_acc)
        if iteration != 1:
            return accs,accs_sum,accs_count13,accs_rank,accs_count26
        return accs

    def train(self, model, train_loader):
        #don't need to train, only show loss
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
        '''
        Run the benchmark wiht designed model.
        '''
        print(f"Using device {self.device}")
        print('Number of nodes in BA graph:',self.NUM_NODES,';Parameter of BA graph:',self.M)
        benchmark_name = self.__class__.__name__
        all_explanations = defaultdict(list)
        all_runtimes = defaultdict(list)
        for experiment_i in tq(range(self.sample_count)):
            # create datasets
            train_dataset = self.create_dataset(self.NUM_TRAIN_GRAPHS, self.M, self.NUM_NODES)
            test_dataset = self.create_dataset(self.NUM_TEST_GRAPHS, self.M, self.NUM_NODES)
            train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size = 1, shuffle = True)
            test_dataloader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size = 1, shuffle = True)
            # set model
            model = GCN_designed(1, 4, 2, False, 'GraphConvWL').to(self.device)
            model.set_paramerters()
            model.to(self.device)
            # test model
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

            # evaluate all explanation methods in "benchmark_dgl.py" files
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


                iteration_num = self.ITERATION_NUM_OF_SUMMARY
                if iteration_num != 1: # use 4 kinds of summarizing methods
                    time_wrapper.explain_function = explain_function
                    accs, accs_sum, accs_count13, accs_rank, accs_count26 = self.evaluate_explanation(time_wrapper, model, test_dataset, explain_name, iteration = iteration_num)
                    print(f'Benchmark:{benchmark_name} Run #{experiment_i + 1}, Explain Method: {explain_name}+{iteration_num}, Accuracy: {np.mean(accs)}')
                    print(f'Benchmark:{benchmark_name} Run #{experiment_i + 1}, Explain Method: {explain_name}+{iteration_num}+sum, Accuracy: {np.mean(accs_sum)}')
                    print(f'Benchmark:{benchmark_name} Run #{experiment_i + 1}, Explain Method: {explain_name}+{iteration_num}+count13, Accuracy: {np.mean(accs_count13)}')
                    print(f'Benchmark:{benchmark_name} Run #{experiment_i + 1}, Explain Method: {explain_name}+{iteration_num}+rank, Accuracy: {np.mean(accs_rank)}')
                    print(f'Benchmark:{benchmark_name} Run #{experiment_i + 1}, Explain Method: {explain_name}+{iteration_num}+count26, Accuracy: {np.mean(accs_count26)}')
                    
                    all_explanations[explain_name].append(list(accs))
                    all_explanations[explain_name+str(iteration_num)+'sum'].append(list(accs_sum))
                    all_explanations[explain_name+str(iteration_num)+'count13'].append(list(accs_count13))
                    all_explanations[explain_name+str(iteration_num)+'rank'].append(list(accs_rank))
                    all_explanations[explain_name+str(iteration_num)+'count26'].append(list(accs_count26))
                    
                    all_runtimes[explain_name].extend(duration_samples)
                    all_runtimes[explain_name+str(iteration_num)+'sum'].extend(duration_samples)
                    all_runtimes[explain_name+str(iteration_num)+'count13'].extend(duration_samples)
                    all_runtimes[explain_name+str(iteration_num)+'rank'].extend(duration_samples)
                    all_runtimes[explain_name+str(iteration_num)+'count26'].extend(duration_samples)

                    metrics = {
                        f'explain_{explain_name}_acc': np.mean(accs),
                        f'time_{explain_name}_{iteration_num}_avg': np.mean(duration_samples),

                        f'explain_{explain_name}_{iteration_num}_sum_acc': np.mean(accs_sum),
                        f'time_{explain_name}_{iteration_num}_sum_avg': np.mean(duration_samples),
                        
                        f'explain_{explain_name}_{iteration_num}_count13_acc': np.mean(accs_count13),
                        f'time_{explain_name}_{iteration_num}_count13_avg': np.mean(duration_samples),
                        
                        f'explain_{explain_name}_{iteration_num}_rank_acc': np.mean(accs_rank),
                        f'time_{explain_name}_{iteration_num}_rank_avg': np.mean(duration_samples),

                        f'explain_{explain_name}_{iteration_num}_count26_acc': np.mean(accs_count26),
                        f'time_{explain_name}_{iteration_num}_count26_avg': np.mean(duration_samples),
                    }
                    with tempfile.TemporaryDirectory() as tmpdir:
                        file_path = os.path.join(tmpdir, 'accuracies.json')
                        json.dump(all_explanations, open(file_path, 'w'), indent=2)
                        mlflow.log_artifact(file_path)
                    mlflow.log_metrics(metrics, step=experiment_i)
                else:# only run explanation method one time
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
