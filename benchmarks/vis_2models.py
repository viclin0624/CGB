# Plot graph with edge mask of same sample when use IG interpret the fixed model and a trained model
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
import matplotlib.pyplot as plt

try:
    model = torch.load('/home/ubuntu/Maolin/eva_gnn/dgl-gnn-exp/benchmarks/result_models/model9.pkl')
except:
    print('Please set the correct path of model.')
model2 = GCN_designed(1, 4, 2, False, 'GraphConvWL')
model2.set_paramerters()
data = CGBDataset(graphs_num=10, m = 5, nodes_num=50, perturb_dic = {}, no_attach_init_nodes = True, include_bias_class=False)
test_dataloader = dgl.dataloading.GraphDataLoader(data, batch_size = 1, shuffle = True)

def draw_explanation(g, edge_mask, name, pos = None):
    # class should in 1,2,3
    if name[2] in ['1','2','3']:
        label = int(name[2])
    else:
        label = int(name[3])
    plt.figure(figsize=(25,25),edgecolor='g')
    fig,ax = plt.subplots(figsize=(10,10))
    g = g.to(torch.device('cpu'))
    g.edata['weight'] = torch.tensor(edge_mask) # edge weights are edge masks
    g_nx = dgl.to_networkx(g,edge_attrs=['weight']) 
    edges = g_nx.edges()
    weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in g_nx.edges(data=True)]
    width = (torch.tensor(edge_mask)-np.min(edge_mask))/np.max(edge_mask)*0.9+0.1
    #set nodes color
    node_color = ['b' for i in range(g.num_nodes())]
    for i in range(1,11):
        node_color[-i] = 'r'
    #set edges color
    edge_color = ['b' for i in range(g.num_edges())]
    ground_truth_num_dic = {1:22, 2:24, 3:26} #number of edges in ground truth with different categories
    '''k = 0 
    for i in np.argsort(-edge_mask):
        if k < ground_truth_num_dic[label]:
            edge_color[i] = 'r'
        #else:
            #width[i] = 0
        k += 1'''
    #for i in range(ground_truth_num_dic[label]-2):
        #edge_color[-(i+1)] = 'r'
    for i in range(g.num_edges()):
        if g.edges()[1][i] in list(range(g.num_nodes()-10, g.num_nodes())) :
            edge_color[i] ='r'
        elif g.edges()[0][i] in list(range(g.num_nodes()-10, g.num_nodes())) and g.edges()[1][i] in list(range(g.num_nodes()-10, g.num_nodes())):
            edge_color[i] = 'r'
    pos = nx.circular_layout(g_nx)
    if pos == None:
        pos = nx.spring_layout(g_nx) #eg: modify position of nodes in class 3
        if name[2] == '3': 
            nodepos = [pos[g.num_nodes()-i] for i in range(1,11)]
            meanx = 0
            meany = 0
            for i in range(10):
                meanx += nodepos[i][0]
                meany += nodepos[i][1]
            meanx = meanx/10
            meany = meany/10
            for i in range(10):
                pos[g.num_nodes()-10+i] = np.array([(nodepos[i][0]-meanx)*1.5+nodepos[i][0],(nodepos[i][1]-meany)*3+nodepos[i][1]])
        if name[3] == '3':
            for i in range(g.num_nodes()):
                if i <40:
                    pos[i] = np.array([0.5+np.random.uniform(6,18),0.5+np.random.uniform(6,18)])
                elif i<45:
                    pos[i] = np.array([0.5+np.random.uniform(0,6),0.5+np.random.uniform(0,6)])
                else:
                    pos[i] = np.array([0.5+np.random.uniform(18,24),0.5+np.random.uniform(18,24)])
    #draw graph
    nx.draw_networkx_nodes(g_nx,pos,node_size=100,node_color=node_color)
    nx.draw_networkx_edges(g_nx,pos,edgelist=weighted_edges,width=width,edge_color=edge_color,connectionstyle="arc3,rad=0.3")
    plt.savefig('visresult/'+name+'.png')
    plt.close()
    return pos
device = torch.device('cpu')#('cuda:0')
count = -1 
model2.to(device)
model.to(device)
for g, label in test_dataloader:
    g = g.to(device)
    count += 1
    edge_mask = explain_ig(model2, 'graph', g, g.ndata['x'], label)
    model_result = model2(g, g.ndata['x'])
    pos = draw_explanation(g, edge_mask, str(count)+'f'+str(label.cpu().item())+str(model_result.detach().cpu().numpy()))

    edge_mask = explain_ig(model, 'graph', g, g.ndata['x'], label)
    model_result = model(g, g.ndata['x'])
    draw_explanation(g, edge_mask, str(count)+'uf'+str(label.cpu().item())+str(model_result.detach().cpu().numpy()),pos)
    plt.xticks(size = 40)
    plt.yticks(size = 40)
    plt.xlim(-4,4)
    plt.hist(edge_mask,bins = 50)
    plt.savefig('visresult/hist'+str(count)+'.png')
    
    