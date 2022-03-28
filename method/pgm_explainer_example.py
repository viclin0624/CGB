import time
from pgmpy import device
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from pgmpy.estimators.CITests import chi_square

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from dgl.nn import GraphConv#instead of GCNConv in PyG
import dgl
import numpy as np

from torch import nn

import networkx as nx

def n_hops_A(A, n_hops):
    # Compute the n-hops adjacency matrix
    adj = torch.tensor(A, dtype=torch.float)
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.numpy().astype(int)

class Graph_Explainer:
    def __init__(
        self,
        model,
        graph,
        num_layers = None,
        perturb_feature_list = None,
        perturb_mode = "mean", # mean, zero, max or uniform
        perturb_indicator = "diff", # diff or abs
    ):
        '''
        Parameters
        ----------
        model: explained model
        graph: explained graph
        num_layers: not use in graph explainer
        perturb_feature_list: which feature to perturb
        perturb_mode: how to perturb
        perturb_indicator: use "diff" or "abs" when calculate and compare change between perturbed data and original data
        '''
        self.model = model
        self.model.eval()
        self.graph = graph
        self.num_layers = num_layers
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        self.X_feat = torch.ones((graph.num_nodes(),1)).numpy()
        self.E_feat = torch.ones((graph.num_edges(),1)).numpy()
    def perturb_features_on_node(self, feature_matrix, node_idx, random = 0):
        '''
        Description
        -----------
        Generate perturb array for one node.

        Parameters
        ----------
        feature_matrix: feature matrix of all nodes
        node_idx: target node index
        random: decide whether random generate perturb array
        '''
        X_perturb = feature_matrix.copy()
        perturb_array = X_perturb[node_idx].copy()
        epsilon = 0.05*np.max(self.X_feat, axis = 0)
        seed = np.random.randint(2)
        
        if random == 1:
            if seed == 1:
                for i in range(perturb_array.shape[0]):
                    if i in self.perturb_feature_list:
                        if self.perturb_mode == "mean":
                            perturb_array[i] = np.mean(feature_matrix[:,i])
                        elif self.perturb_mode == "zero":
                            perturb_array[i] = 0
                        elif self.perturb_mode == "max":
                            perturb_array[i] = np.max(feature_matrix[:,i])
                        elif self.perturb_mode == "uniform":
                            perturb_array[i] = perturb_array[i] + np.random.uniform(low=-epsilon[i], high=epsilon[i])
                            if perturb_array[i] < 0:
                                perturb_array[i] = 0
                            elif perturb_array[i] > np.max(self.X_feat, axis = 0)[i]:
                                perturb_array[i] = np.max(self.X_feat, axis = 0)[i]

        
        X_perturb[node_idx] = perturb_array

        return X_perturb 
    
    def batch_perturb_features_on_node(self, num_samples, index_to_perturb,
                                            percentage, p_threshold, pred_threshold):
        '''
        Description
        -----------
        Generate a batch samples perturbed.

        Parameters
        ----------
        num_samples: num of samples for every node
        index_to_perturb: which nodes are perturbed
        percentage: percentage of nodes actually perturbed in index_to_perturb
        p_threshold, pred_threshold: Not used in this function
        '''
        X_torch = torch.tensor(self.X_feat).to(next(self.model.parameters()).device)
        E_torch = torch.tensor(self.E_feat).to(next(self.model.parameters()).device)
        #Calculate original output
        pred_torch = self.model.forward(self.graph, X_torch, E_torch).cpu()
        soft_pred = np.asarray(pred_torch[0].data)
        pred_label = np.argmax(soft_pred)
        num_nodes = self.X_feat.shape[0]
        Samples = [] 
        for iteration in range(num_samples):
            #Generate perturb features
            X_perturb = self.X_feat.copy()
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(X_perturb, node, random = latent)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)
            #Calculate change of probability output
            X_perturb_torch =  torch.tensor(X_perturb, dtype=torch.float).to(next(self.model.parameters()).device)
            pred_perturb_torch = self.model.forward(self.graph, X_perturb_torch, E_torch).cpu()
            soft_pred_perturb = np.asarray(pred_perturb_torch[0].data)

            pred_change = np.max(soft_pred) - soft_pred_perturb[pred_label]
            
            #Add pred_change to "sample" as a new node indexed "num_nodes"
            sample.append(pred_change)
            Samples.append(sample)
        
        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)
        #Set label of top 1/8 samples which change the most to 1
        top = int(num_samples/8)
        top_idx = np.argsort(Samples[:,num_nodes])[-top:] 
        for i in range(num_samples):
            if i in top_idx:
                Samples[i,num_nodes] = 1
            else:
                Samples[i,num_nodes] = 0
            
        return Samples
    
    def explain(self, num_samples = 10, percentage = 50, top_node = None, p_threshold = 0.05, pred_threshold = 0.1):
        '''

        Return explain result with selected nodes, p-value of all nodes, candidate nodes selected in Round 1.
        
        Parameters
        ----------
        num_samples: num of samples in Round 2
        percentage: use in function batch_perturb_features_on_node as percentage of nodes actually perturbed in index_to_perturb
        top_node: output num of nodes
        p_threshold, pred_threshold: not use in this function
        '''
        num_nodes = self.X_feat.shape[0]
        if top_node == None:
            top_node = int(num_nodes/20)
        
        #Round 1 generate perturb data and first select candidate nodes
        Samples = self.batch_perturb_features_on_node(int(num_samples/2), range(num_nodes),percentage, 
                                                            p_threshold, pred_threshold)         
        
        data = pd.DataFrame(Samples)
        
        p_values = []
        candidate_nodes = []
        
        #Use chi_square test to select nodes
        target = num_nodes # The entry for the graph classification data is at "num_nodes"
        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
        
        number_candidates = int(top_node*4)
        candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]
        
        #Round 2 generate perturb data again but only perturb candidate nodes in Round 1 and second select nodes as result
        Samples = self.batch_perturb_features_on_node(num_samples, candidate_nodes, percentage, 
                                                            p_threshold, pred_threshold)          
        data = pd.DataFrame(Samples)
      
        p_values = []
        dependent_nodes = []
        
        #Use chi_square test again to select nodes whose p-value is less as result
        target = num_nodes
        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)

        top_p = np.min((top_node,num_nodes-1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)
        
        return pgm_nodes, p_values, candidate_nodes

class GraphConvWL(nn.Module):
    r'''
    Description
    -----------
    Similar to GraphConv in PyG

    This graph convolution operater was introduced in `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    bias : bool, optional
        If True, apply a learnable bias to the output. Default: ``True``
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``True``.
    '''
    def __init__(self,
                in_feats,
                out_feats,
                bias = True,
                allow_zero_in_degree = True) -> None:
        super().__init__()
        self.conv_from_neigh = GraphConv(in_feats, out_feats, norm = 'none', weight = True, bias = bias, allow_zero_in_degree = allow_zero_in_degree)
        self.conv_from_self = nn.Linear(in_feats,out_feats, bias = False)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv_from_neigh.reset_parameters()
        self.conv_from_self.reset_parameters()

    def forward(self, graph, feat, edge_weight = None):
        '''
        Parameters
        ----------
        graph: DGL Graph
        feat: node features
        edge_weight: edge weight is necessary for PGMExplainer
        '''
        neigh_feat = self.conv_from_neigh(graph, feat, edge_weight = edge_weight)
        self_feat = self.conv_from_self(feat)
        return neigh_feat+self_feat

class FixedNet2(nn.Module):
    def __init__(self, num_node_features, num_classes, num_layers, dim):
        '''
        Parameters
        ----------
        num_node_features: dim of feature
        num_classes: num of class
        num_layers: num of layers
        dim: dim of hidden layers
        '''
        super(FixedNet2, self).__init__()
        self.convs = torch.nn.ModuleList()
        conv_class = GraphConvWL
        self.convs.append(conv_class(num_node_features, dim))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim, bias = True))

        self.fc1 = Linear(dim, num_classes, bias = True)


    def forward(self, g, x, edge_weight = None):
        '''
        Parameters
        ----------
        g: DGL Graph
        x: node feature
        edge_weight: edge weight is necessary for PGMExplainer
        '''
        for conv in self.convs:
            x = conv(g, x, edge_weight)
            x = F.relu(x)
        g.ndata['h'] = x
        hg = dgl.sum_nodes(g, 'h')
        output = self.fc1(hg)
        return F.softmax(output, dim = 1)

def node_attr_to_edge(g, node_mask):
    #Convert node mask to edge mask
    edge_mask = np.zeros(g.num_edges())
    edge_mask += node_mask[g.edges()[0].cpu().numpy()]
    edge_mask += node_mask[g.edges()[1].cpu().numpy()]
    return edge_mask

if __name__ == '__main__':
    #Set model and example data
    model = FixedNet2(1,2,2,8)
    g = dgl.graph((torch.tensor([i for i in range(50)]),torch.tensor([i for i in range(1,51)])))
    x = torch.ones((g.num_nodes(),1))

    #Get pred_threshold by predict of model (Actually not use in Explainer)
    pred = model.forward(g, x)
    soft_pred = np.array(pred[0].data)
    pred_threshold = 0.1*np.max(soft_pred)

    #Implement Graph_Explainer with 
    e = Graph_Explainer(model, g,
                            perturb_feature_list = [0],
                            perturb_mode = "mean",
                            perturb_indicator = "diff")
    pgm_nodes, p_values, candidates = e.explain(num_samples = 1000, percentage = 10, 
                            top_node = 5, p_threshold = 0.05, pred_threshold = pred_threshold)
    explanation = zip(pgm_nodes,p_values)

    #Importance of node = 1 - p-value, convert node importance to edge importance
    node_attr = np.zeros(x.shape[0])
    for node, p_value in explanation:
        node_attr[node] = 1 - p_value
    edge_mask = node_attr_to_edge(g, node_attr)
    print(edge_mask)