
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
#from torch_geometric.nn import GNNExplainer, GINConv, MessagePassing, GCNConv, GraphConv
from dgl.nn import GraphConv#instead of GCNConv in PyG
import dgl
import numpy as np

from torch import nn
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
        neigh_feat = self.conv_from_neigh(graph, feat, edge_weight = edge_weight)
        self_feat = self.conv_from_self(feat)
        return neigh_feat+self_feat


class Net1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type):
        super(Net1, self).__init__()
        dim = 32
        self.convs = torch.nn.ModuleList()
        if conv_type == 'GraphConvWL':#'GCNConv':
            conv_class = GraphConvWL
            #kwargs = {'add_self_loops': False}
        elif conv_type == 'GraphConv':
            conv_class = GraphConv
            kwargs = {}
        else:
            raise RuntimeError(f"conv_type {conv_type} not supported")

        self.convs.append(conv_class(num_node_features, dim))#, **kwargs))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim))#, **kwargs))
        self.concat_features = concat_features
        if concat_features:
            self.fc = Linear(dim * num_layers + num_node_features, num_classes)
        else:
            self.fc = Linear(dim, num_classes)

    def forward(self, g, x):
        '''
        g: DGL Graph
        x: node feature
        '''
        xs = [x]
        for conv in self.convs:
            x = conv(g, x)
            x = F.relu(x)
            xs.append(x)
        if self.concat_features:
            x = torch.cat(xs, dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

        
class Net2(torch.nn.Module):
    '''
    For graph classification
    '''
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type, readout = 'Sum'):
        super(Net2, self).__init__()
        dim = 16
        self.convs = torch.nn.ModuleList()
        self.readout = readout 
        if conv_type == 'GraphConvWL':#'GCNConv':
            conv_class = GraphConvWL
            #kwargs = {'add_self_loops': False}
        elif conv_type == 'GraphConv':
            conv_class = GraphConv
            kwargs = {}
        else:
            raise RuntimeError(f"conv_type {conv_type} not supported")

        self.convs.append(conv_class(num_node_features, dim))#, **kwargs))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim))#, **kwargs))
        self.concat_features = concat_features
        if concat_features:
            self.fc = Linear(dim * num_layers + num_node_features, num_classes)
        else:
            self.fc = Linear(dim, num_classes)

    def forward(self, g, x, edge_weight = None):
        '''
        g: DGL Graph
        x: node feature
        '''
        xs = [x]
        for conv in self.convs:
            x = conv(g, x, edge_weight)
            x = F.relu(x)
            xs.append(x)
        if self.concat_features:
            x = torch.cat(xs, dim=1)
        g.ndata['h'] = x
        if self.readout == 'Mean':
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == 'Max':
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == 'Sum':
            hg = dgl.sum_nodes(g, 'h')
        hg = self.fc(hg)
        return F.log_softmax(hg, dim=1)

class FixedNet(nn.Module):
    '''
    Control parameters in this model and can use use_report and unuse_report to set if report every output of layers.
    '''
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type, report = False):
        super(FixedNet, self).__init__()
        dim = 1
        self.report = report
        self.convs = torch.nn.ModuleList()
        if conv_type == 'GraphConvWL':#'GCNConv':
            conv_class = GraphConvWL
            #kwargs = {'add_self_loops': False}
        elif conv_type == 'GraphConv':
            conv_class = GraphConv
            kwargs = {}
        else:
            raise RuntimeError(f"conv_type {conv_type} not supported")

        self.convs.append(conv_class(num_node_features, dim, bias = True))#, **kwargs))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim, bias = True))#, **kwargs))
        self.concat_features = concat_features


    def forward(self, g, x):
        '''
        g: DGL Graph
        x: node feature
        '''
        xs = [x]
        for conv in self.convs:
            x = conv(g, x)
            x = F.relu(x)
            xs.append(x)
        if self.concat_features:
            x = torch.cat(xs, dim=1)
        g.ndata['h'] = x
        hg = dgl.sum_nodes(g, 'h')
        if self.report == True:
            return hg, xs
        '''result = []
        for h in hg:
            if h == 0:
                result.append([1,0,0,0])
            elif h == 20:
                result.append([0,1,0,0])
            elif h == 14:
                result.append([0,0,1,0])
            else:
                result.append([0,0,0,1])
        return torch.tensor(result)'''
        return hg
    
    def use_report(self):
        self.report = True

    def unuse_report(self):
        self.report = False

    def set_paramerters(self):
        k = 0
        for p in self.parameters():
            if k == 0 or k == 2:
                torch.nn.init.constant_(p, 1)
            elif k == 3:
                torch.nn.init.constant_(p, -1)
            elif k == 5:
                torch.nn.init.constant_(p, 3)
            else:
                torch.nn.init.constant_(p, 0)
            k += 1


class FixedNet2(nn.Module):
    '''
    Control parameters in this model and can use use_report and unuse_report to set if report every output of layers.
    '''
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type, report = False):
        super(FixedNet2, self).__init__()
        dim = 1
        self.report = report
        self.convs = torch.nn.ModuleList()
        if conv_type == 'GraphConvWL':#'GCNConv':
            conv_class = GraphConvWL
            #kwargs = {'add_self_loops': False}
        elif conv_type == 'GraphConv':
            conv_class = GraphConv
            kwargs = {}
        else:
            raise RuntimeError(f"conv_type {conv_type} not supported")

        self.convs.append(conv_class(num_node_features, dim, bias = True))#, **kwargs))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim, bias = True))#, **kwargs))
        self.concat_features = concat_features

        self.fc1 = Linear(1,8, bias = True)
        self.output = Linear(8,4, bias = True)



    def forward(self, g, x, edge_weight = None):
        '''
        g: DGL Graph
        x: node feature
        '''
        xs = [x]
        for conv in self.convs:
            x = conv(g, x, edge_weight)
            x = F.relu(x)
            xs.append(x)
        if self.concat_features:
            x = torch.cat(xs, dim=1)
        g.ndata['h'] = x
        hg = dgl.sum_nodes(g, 'h')
        hg2 = self.fc1(hg)
        hg2 = torch.sigmoid(hg2*1000)
        output = self.output(hg2)
        output = F.relu(output)
        return F.log_softmax(output, dim = 1)
    
    def use_report(self):
        self.report = True

    def unuse_report(self):
        self.report = False

    def set_paramerters(self):
        k = 0
        for p in self.parameters():
            if k == 0 or k == 2:
                torch.nn.init.constant_(p, 1)
            elif k == 3:
                torch.nn.init.constant_(p, -1)
            elif k == 5:
                torch.nn.init.constant_(p, 3)
            elif k == 1 or k == 4:
                torch.nn.init.constant_(p, 0)
            elif k == 6: #W in fc1
                with torch.no_grad():
                    temp = [1,-1]
                    for _ in range(2):
                        temp.extend(temp)
                    temp = torch.tensor(temp,dtype = torch.float32)
                    temp = temp.reshape((-1,1))
                    self.fc1.weight = torch.nn.Parameter(temp)
            elif k == 7: #Bias in fc1
                with torch.no_grad():
                    temp = torch.tensor([0.01,0.01, -19.99,20.01, -13.99,14.01, -7.99,8.01],dtype = torch.float32)
                    self.fc1.bias = torch.nn.Parameter(temp)
            elif k == 8: #W in fc2
                with torch.no_grad():
                    temp = torch.zeros((4,8),dtype = torch.float32)
                    for i in range(4):
                        temp[i,i*2] = 100
                        temp[i,i*2+1] = 100 
                    self.output.weight = torch.nn.Parameter(temp)
            elif k == 9: #Bias in fc2
                with torch.no_grad():
                    temp = torch.ones(4,dtype = torch.float32) * -100
                    self.output.bias = torch.nn.Parameter(temp)
            k += 1

    def set_random_gnn_parameters(self, low = -0.0001, high = 0.0001):
        k = 0
        for p in self.parameters():
            if k == 0 or k == 2:
                torch.nn.init.constant_(p, p[0].item()+np.random.uniform(low,high))
            elif k == 3:
                torch.nn.init.constant_(p, p.item()+np.random.uniform(low,high))
            elif k == 5:
                torch.nn.init.constant_(p, p.item()+np.random.uniform(low,high))
            elif k == 1 or k == 4:
                torch.nn.init.constant_(p, p.item()+np.random.uniform(low,high))
            k += 1

class GCN_train(nn.Module):
    '''
    Control parameters in this model and can use use_report and unuse_report to set if report every output of layers.
    '''
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type, report = False):
        super(GCN_train, self).__init__()
        dim = 32
        self.report = report
        self.convs = torch.nn.ModuleList()
        if conv_type == 'GraphConvWL':#'GCNConv':
            conv_class = GraphConvWL
            #kwargs = {'add_self_loops': False}
        elif conv_type == 'GraphConv':
            conv_class = GraphConv
            kwargs = {}
        else:
            raise RuntimeError(f"conv_type {conv_type} not supported")

        self.convs.append(conv_class(num_node_features, dim, bias = True))#, **kwargs))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim, bias = True))#, **kwargs))
        self.concat_features = concat_features

        self.fc1 = Linear(dim,8, bias = True)
        self.output = Linear(8,4, bias = True)



    def forward(self, g, x, edge_weight = None):
        '''
        g: DGL Graph
        x: node feature
        '''
        xs = [x]
        for conv in self.convs:
            x = conv(g, x, edge_weight)
            x = F.relu(x)
            xs.append(x)
        if self.concat_features:
            x = torch.cat(xs, dim=1)
        g.ndata['h'] = x
        hg = dgl.sum_nodes(g, 'h')
        hg2 = self.fc1(hg)
        hg2 = F.relu(hg2)
        output = self.output(hg2)
        output = F.relu(output)
        return F.softmax(output, dim = 1)
    