
from curses import start_color
from pickletools import int4
import random
from collections import defaultdict

import json
import os
import tempfile
import time
from utils import record_exp

import mlflow
import networkx as nx
import numpy as np
import torch
#from torch_geometric.utils import from_networkx
import dgl

import sys
sys.path.append("..")
from model.models_dgl import Net2

from tqdm import tqdm as tq

from benchmark_dgl import Benchmark

import math
import torch.nn.functional as F

from dgl.data import DGLDataset
def ba(start, width, role_start=0, m=5):
    """Builds a BA preferential attachment graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int size of the graph
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    return graph, roles

def house(start, role_start=0):
    """Builds a house-like  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                    role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    return graph, roles

def square_diagonal(start, role_start=0):
    """Builds a square_diagonal  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                    role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 4))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start, start + 2)])
    roles = [role_start, role_start, role_start, role_start]
    return graph, roles



def five_cycle(start, role_start = 0):
    """Builds a five-cycle  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                    role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start + 4),
            (start + 4, start)
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    #graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    return graph, roles

def three_cycle(start, role_start = 0):
    """Builds a three-cycle  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                    role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 3))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start)
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    #graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start]
    return graph, roles
def four_cycle(start, role_start = 0):
    """Builds a four-cycle  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                    role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 4))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start)
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    #graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start, role_start]
    return graph, roles
def six_cycle(start, role_start = 0):
    """Builds a six-cycle  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                    role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 6))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start + 4),
            (start + 4, start + 5),
            (start + 5, start)
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    #graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start, role_start, role_start, role_start]
    return graph, roles
def build_graph(
    width_basis,
    basis_type,
    list_shapes,
    start=0,
    rdm_basis_plugins=False,
    add_random_edges=0,
    m=5,
):
    """This function creates a basis (scale-free, path, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    m                :      number of edges to attach to existing node (for BA graph)
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    plugins          :      node ids with the attached shapes
    """
    if basis_type == "ba":
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis)

    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node
    
    if n_shapes != 0:
    # Sample (with replacement) where to attach the new motifs
        if rdm_basis_plugins is True:
            plugins = np.random.choice(n_basis, n_shapes, replace=False)
        else:
            spacing = math.floor(n_basis / n_shapes)
            plugins = [int(k * spacing) for k in range(n_shapes)]
        seen_shapes = {"basis": [0, n_basis]}

        for shape_id, shape in enumerate(list_shapes):
            shape_type = shape[0]
            args = [start]
            if len(shape) > 1:
                args += shape[1:]
            args += [0]
            graph_s, roles_graph_s = eval(shape_type)(*args)
            n_s = nx.number_of_nodes(graph_s)
            try:
                col_start = seen_shapes[shape_type][0]
            except:
                col_start = np.max(role_id) + 1
                seen_shapes[shape_type] = [col_start, n_s]
            # Attach the shape to the basis
            basis.add_nodes_from(graph_s.nodes())
            basis.add_edges_from(graph_s.edges())
            basis.add_edges_from([(start, plugins[shape_id])])
            if shape_type == "cycle":
                if np.random.random() > 0.5:
                    a = np.random.randint(1, 4)
                    b = np.random.randint(1, 4)
                    basis.add_edges_from([(a + start, b + plugins[shape_id])])
            temp_labels = [r + col_start for r in roles_graph_s]
            # temp_labels[0] += 100 * seen_shapes[shape_type][0]
            role_id += temp_labels
            start += n_s

        if add_random_edges > 0:
            # add random edges between nodes:
            for p in range(add_random_edges):
                src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
                print(src, dest)
                basis.add_edges_from([(src, dest)])

        return basis, role_id, plugins
    else:
        return basis, role_id, []

class BA4labelDataset(DGLDataset):
    basis_type = "ba"

    def __init__(self, graphs_num = 1000, nodes_num = 25, m = 1):
        self.graphs_num = graphs_num
        self.nodes_num = nodes_num
        self.m = m
        super(BA4labelDataset, self).__init__('BA4labelDataset')

    def process(self):
        self.graphs = []
        self.labels = []
        self.role_id = []
        self.plug_id = []
        #self.pertube_dic = {4:'square_diagonal'}
        self.pertube_dic = {3:'three_cycle', 4:'four_cycle', 6:'six_cycle'}
        for _ in range(self.graphs_num):
            which_type = np.random.choice([0,1,2,3])
            perturb_type = np.random.choice([0,3,4,6])
            if which_type == 0:
                if perturb_type == 0:
                    G, role_id, plug_id = build_graph(self.nodes_num, self.basis_type, [], start = 0, m = self.m)
                else:
                    G, role_id, plug_id = build_graph(self.nodes_num - perturb_type, self.basis_type, [[self.pertube_dic[perturb_type]]], start = 0, m = self.m)
            else:
                list_shapes = [["house"]] * (which_type - 1) + [["five_cycle"]] * (3 - which_type)
                if perturb_type != 0:
                    list_shapes = list_shapes + [[self.pertube_dic[perturb_type]]]
                G, role_id, plug_id = build_graph(self.nodes_num-10-perturb_type, self.basis_type, list_shapes, start = 0, m = self.m)
            
            g = dgl.from_networkx(G)
            g.ndata['x'] = torch.ones((self.nodes_num,1))
            self.graphs.append(g)
            self.labels.append(which_type)
            self.role_id.append(role_id)
            self.plug_id.append(plug_id)

    @property
    def num_classes(self):
        return 4

    @property
    def num_node_features(self):
        return 1


    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


class BA4label(Benchmark):
    NUM_GRAPHS = 10
    TEST_RATIO = 0.4
    LR = 0.005

    @staticmethod
    def get_accuracy(correct_ids, edge_mask, edge_index):
        '''
        edge_index: 2 elements tuple, u and v
        '''

        correct_count = 0
        correct_edges = list(zip(correct_ids, correct_ids[1:]))

        for x in np.argsort(-edge_mask)[:len(correct_ids)]:
            u = edge_index[0][x]
            v = edge_index[1][x]
            u, v = u.item(), v.item()
            if (u, v) in correct_edges:
                correct_count += 1
        return correct_count / len(correct_edges)

    @staticmethod
    def get_accuracy_undirected(correct_ids, edge_values):
        correct_count = 0
        correct_edges = list(zip(correct_ids, correct_ids[1:]))

        top_edges = list(sorted([(-value, edge) for edge, value in edge_values.items()]))[:len(correct_ids)]
        for _, (u, v) in top_edges:
            if (u, v) in correct_edges or (v, u) in correct_edges:
                correct_count += 1
        return correct_count / len(correct_edges)

    def create_dataset(self):
        '''
        Return data with 
        '''

        basis_type = "ba"
        which_type = np.random.choice([0,1,2,3])
        if which_type == 0:
            G, role_id, plug_id = build_graph(25, basis_type, [], start = 0, m = 1)
        else:
            list_shapes = [["house"]] * (which_type - 1) + [["five_cycle"]] * (3 - which_type)
            G, role_id, plug_id = build_graph(15, basis_type, list_shapes, start = 0, m = 1)
        
        data = dgl.from_networkx(G)
        data.role_id = role_id 
        data.plug_id = plug_id
        data.y = torch.tensor([which_type], dtype = torch.int8)
        data.num_classes = 4
        data.ndata['x'] = torch.ones((data.num_nodes(),1))#torch.tensor([1 for _ in range(data.num_nodes())])
        data.num_node_features = data.ndata['x'].shape[1]
        print('created one')
        return data

    def is_trained_model_valid(self, test_acc):
        return test_acc > 0.999

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name):
        accs = []
        misclassify_count = 0
        for data in test_dataset:
            data = data.to(self.device)
            _, pred = model(data, data.ndata['x']).max(dim=1)#data.x data.edge_index
            nodes_to_test = list(zip(data.unique_solution_nodes, data.unique_solution_explanations))
            nodes_to_test = self.subsample_nodes(explain_function, nodes_to_test)
            pbar = tq(nodes_to_test, disable=False)
            tested_nodes = 0
            for node_idx, correct_ids in pbar:
                if pred[node_idx] != data.ndata['y'][node_idx]:
                    misclassify_count += 1
                    continue
                tested_nodes += 1
                edge_mask = explain_function(model, node_idx, data.ndata['x'], data.edges(), data.ndata['y'][node_idx].item())
                explain_acc = self.get_accuracy(correct_ids, edge_mask, data.edges())
                accs.append(explain_acc)
                pbar.set_postfix(acc=np.mean(accs))
            mlflow.log_metric('tested_nodes_per_graph', tested_nodes)
        return accs

    def train(self, model, optimizer, train_loader):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data, data.ndata['x'])
            loss = F.nll_loss(output, data.y)
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
            correct += pred.eq(data.y).sum().item()
            total += len(data.y)
        return correct / total


    def run(self):
        print(f"Using device {self.device}")
        benchmark_name = self.__class__.__name__
        all_explanations = defaultdict(list)
        all_runtimes = defaultdict(list)
        for experiment_i in tq(range(self.sample_count)):
            dataset = [self.create_dataset() for i in range(self.NUM_GRAPHS)]
            split_point = int(len(dataset) * self.TEST_RATIO)
            test_dataset = dataset[:split_point]
            train_dataset = dataset[split_point:]
            data = dataset[0]
            model = Net2(data.num_node_features, data.num_classes, self.num_layers, self.concat_features,
                         self.conv_type).to(
                self.device)
            train_acc, test_acc = self.train_and_test(model, train_dataset, test_dataset)
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
                
if __name__ == '__main__': 
    result_dic = {}
    for m in tq([1,5]):
        for layers in tq([1,2,3,4]):
            data = BA4labelDataset(m = m,nodes_num=50)
            dataloader = dgl.dataloading.GraphDataLoader(data, batch_size = 16, shuffle = True)
            data2 = BA4labelDataset(m = m,nodes_num=50)
            testdataloader = dgl.dataloading.GraphDataLoader(data2, batch_size = 16, shuffle = True)
            device = torch.device('cuda')
            
            total_train_acc = []
            total_test_acc = []
            train_acc_dic = {}
            test_acc_dic = {}
            for i in range(4):
                train_acc_dic[i] = [0,0,0,0,0,0] #correct total 0 1 2 3
                test_acc_dic[i] = [0,0,0,0,0,0] #correct total 0 1 2 3
            for _ in tq(range(10)):
                model = Net2(data.num_node_features, data.num_classes, layers, True,
                                    'GraphConvWL').to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0)
                
                model.train()
                pbar = tq(range(150))
                for epoch in pbar:
                    #train
                    loss_all = 0
                    for g, labels in dataloader:
                        g = g.to(device)
                        optimizer.zero_grad()
                        output = model(g, g.ndata['x'])
                        loss = F.nll_loss(output, labels.to(device))
                        loss.backward()
                        loss_all += loss.item()
                        optimizer.step()
                    train_loss = loss_all/len(dataloader)
                    #train_acc
                    model.eval()

                    correct = 0
                    total = 0
                    for g, labels in dataloader:
                        g = g.to(device)
                        output = model(g, g.ndata['x'])
                        pred = output.max(dim=1)[1]
                        eq_pred = pred.eq(labels.to(device))
                        correct += eq_pred.sum().item()
                        for index,label in enumerate(labels.tolist()):
                            train_acc_dic[label][0] = train_acc_dic[label][0] + eq_pred[index].item()
                            train_acc_dic[label][1] = train_acc_dic[label][1] + 1
                            train_acc_dic[label][2] = train_acc_dic[label][2] + int(pred[index] == 0)
                            train_acc_dic[label][3] = train_acc_dic[label][3] + int(pred[index] == 1)
                            train_acc_dic[label][4] = train_acc_dic[label][4] + int(pred[index] == 2)
                            train_acc_dic[label][5] = train_acc_dic[label][5] + int(pred[index] == 3)
                        total += len(labels.to(device))
                    train_acc = correct/total
                    #test_acc
                    model.eval()

                    correct = 0
                    total = 0
                    for g, labels in testdataloader:
                        g = g.to(device)
                        output = model(g, g.ndata['x'])
                        pred = output.max(dim=1)[1]
                        eq_pred = pred.eq(labels.to(device))
                        correct += eq_pred.sum().item()
                        for index,label in enumerate(labels.tolist()):
                            test_acc_dic[label][0] = test_acc_dic[label][0] + eq_pred[index].item()
                            test_acc_dic[label][1] = test_acc_dic[label][1] + 1
                            test_acc_dic[label][2] = test_acc_dic[label][2] + int(pred[index] == 0)
                            test_acc_dic[label][3] = test_acc_dic[label][3] + int(pred[index] == 1)
                            test_acc_dic[label][4] = test_acc_dic[label][4] + int(pred[index] == 2)
                            test_acc_dic[label][5] = test_acc_dic[label][5] + int(pred[index] == 3)
                        total += len(labels.to(device))
                    test_acc = correct/total

                    pbar.set_postfix(train_loss=train_loss, train_acc = train_acc, test_acc = test_acc)
                total_train_acc.append(train_acc)
                total_test_acc.append(test_acc)
            print(np.mean(total_train_acc))
            print(np.mean(total_test_acc))
            print(train_acc_dic, test_acc_dic)
            result_dic[(m, layers)] = (np.mean(total_train_acc), np.mean(total_test_acc), train_acc_dic, test_acc_dic)
    print(result_dic)
    record_exp(result_dic, 5)
    #A = BA4label(10,4,True,'GraphConvWL')
    #A.run()
