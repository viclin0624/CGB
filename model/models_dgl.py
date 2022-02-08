
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
#from torch_geometric.nn import GNNExplainer, GINConv, MessagePassing, GCNConv, GraphConv
from dgl.nn import GraphConv#instead of GCNConv in PyG
import dgl

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

    def forward(self, graph, feat):
        neigh_feat = self.conv_from_neigh(graph, feat)
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
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type, readout = 'Mean'):
        super(Net2, self).__init__()
        dim = 32
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
        if self.readout == 'Mean':
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == 'Max':
            hg = dgl.max_nodes(g, 'h')
        hg = self.fc(hg)
        return F.log_softmax(hg, dim=1)