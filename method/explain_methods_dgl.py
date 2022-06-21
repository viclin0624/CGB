from typing import Union, Tuple, Any
from scipy.special import softmax

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_additional_forward_args,
    _format_input,
    _format_output,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType
from captum.attr import Saliency, IntegratedGradients, LayerGradCam
from torch import Tensor
#from torch_geometric.data import Data
import dgl
from method.NodeExplainerModule import NodeExplainerModule
from method.gnn_explainer import GNNExplainer
import method.pgm_explainer as pe
from torch.utils.data import DataLoader
#from torch_geometric.nn import MessagePassing
#from torch_geometric.utils import to_networkx

#from pgm_explainer import Node_Explainer
#from gnn_explainer import TargetedGNNExplainer

#from benchmarks.subgraphx import SubgraphX,MCTS
#from benchmarks.subgraphx import find_closest_node_result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphLayerGradCam(LayerGradCam):

    def attribute(self, inputs: Union[Tensor, Tuple[Tensor, ...]], target: TargetType = None,
                  additional_forward_args: Any = None, attribute_to_layer_input: bool = False,
                  relu_attributions: bool = False) -> Union[Tensor, Tuple[Tensor, ...]]:
        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)

        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=0,
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)
        return _format_output(len(scaled_acts) > 1, scaled_acts)


def model_forward(input_mask, g, model, x):
    if input_mask.shape[0] != g.num_edges():
        out = []
        for i in range(int(input_mask.shape[0]/g.num_edges())):
            out.append(model(g, x[(i*g.num_nodes()):((i+1)*g.num_nodes())], input_mask[(i*g.num_edges()):((i+1)*g.num_edges())]))
        out = torch.cat(out, dim = 0)
    else:
        out = model(g, x, input_mask)
    return out


def model_forward_node(g, model, x, node_idx):
    out = model(g, x)
    return out[[node_idx]]


def node_attr_to_edge(g, node_mask):
    edge_mask = np.zeros(g.num_edges())
    edge_mask += node_mask[g.edges()[0].cpu().numpy()]
    edge_mask += node_mask[g.edges()[1].cpu().numpy()]
    return edge_mask


def get_all_convolution_layers(model):
    layers = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            layers.append(module)
    return layers

def collate(self, samples):
    # The input samples is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels))
    tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
    tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
    snorm_n = torch.cat(tab_snorm_n).sqrt()  
    tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
    tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
    snorm_e = torch.cat(tab_snorm_e).sqrt()
    for idx, graph in enumerate(graphs):
        graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
        graphs[idx].edata['feat'] = graph.edata['feat'].float()
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels, snorm_n, snorm_e

def explain_random(model, task_type, g, x, target):
    return np.random.uniform(size=g.num_edges())


def explain_gradXact(model, node_idx, x, edge_index, target, include_edges=None):
    # Captum default implementation of LayerGradCam does not average over nodes for different channels because of
    # different assumptions on tensor shapes
    input_mask = x.clone().requires_grad_(True).to(device)
    layers = get_all_convolution_layers(model)
    node_attrs = []
    for layer in layers:
        layer_gc = LayerGradCam(model_forward_node, layer)
        node_attr = layer_gc.attribute(input_mask, target=target, additional_forward_args=(model, edge_index, node_idx))
        node_attr = node_attr.cpu().detach().numpy().ravel()
        node_attrs.append(node_attr)
    node_attr = np.array(node_attrs).mean(axis=0)
    edge_mask = node_attr_to_edge(edge_index, node_attr)
    return edge_mask



def explain_sa(model, task_type, g, x, target):
    saliency = Saliency(model_forward)
    input_mask = torch.ones(g.num_edges()).requires_grad_(True).to(device)
    input_mask.retain_grad()
    attr = saliency.attribute(input_mask, target=int(target), additional_forward_args = (g,model,x), abs = False)#IG忽略了边之间的互相影响每次会给所有边相同的weight
    attr = attr.detach().cpu().numpy()
    return attr


def explain_ig(model, task_type, g, x, target):
    ig = IntegratedGradients(model_forward)
    input_mask = torch.ones(g.num_edges()).requires_grad_(True).to(device)
    attr,delta = ig.attribute(input_mask, target=int(target), additional_forward_args = (g,model,x),return_convergence_delta=True, n_steps=500)#IG忽略了边之间的互相影响每次会给所有边相同的weight
    attr = attr.detach().cpu().numpy()
    return attr


def explain_occlusion(model, node_idx, x, edge_index, target, include_edges=None):
    depth_limit = len(model.convs) + 1
    data = Data(x=x, edge_index=edge_index)
    pred_prob = model(data.x, data.edge_index)[node_idx][target].item()
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, target=node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        if include_edges is not None and not include_edges[i].item():
            continue
        u, v = list(edge_index_numpy[:, i])
        if (u, v) in subgraph.edges():
            edge_occlusion_mask[i] = False
            prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][target].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask


def explain_occlusion_undirected(model, node_idx, x, edge_index, target, include_edges=None):
    depth_limit = len(model.convs) + 1
    data = Data(x=x, edge_index=edge_index)
    pred_prob = model(data.x, data.edge_index)[node_idx][target].item()
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    reverse_edge_map = {}
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        u, v = list(edge_index_numpy[:, i])
        reverse_edge_map[(u, v)] = i

    for (u, v) in subgraph.edges():
        if u > v:  # process each edge once
            continue
        i1 = reverse_edge_map[(u, v)]
        i2 = reverse_edge_map[(v, u)]
        if include_edges is not None and not include_edges[i1].item() and not include_edges[i2].item():
            continue
        edge_occlusion_mask[[i1, i2]] = False
        prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][target].item()
        edge_mask[[i1, i2]] = pred_prob - prob
        edge_occlusion_mask[[i1, i2]] = True
    return edge_mask


def explain_gnnexplainer(model, task_type, g, x, target):
    '''# load graph, feat, and label
    labels = target
    feats = x
    dummy_model = model
    num_classes = 4
    feat_dim = feats.shape[1]
    lr = 0.01
    wd = 0
    epochs = 200

    # create an explainer
    explainer = NodeExplainerModule(model=model,
                                    num_edges=g.number_of_edges(),
                                    node_feat_dim=feat_dim)

    # define optimizer
    optim = torch.optim.Adam(explainer.parameters(), lr=lr, weight_decay=wd)

    # train the explainer for the given node
    dummy_model.eval()
    model_logits = dummy_model(g, x)
    model_predict = F.one_hot(torch.argmax(model_logits, dim=-1), num_classes)

    for epoch in range(epochs):
        explainer.train()
        exp_logits = explainer(g, x)
        loss = explainer._loss(exp_logits, model_predict)

        optim.zero_grad()
        loss.backward()
        optim.step()
    edge_weights = explainer.edge_mask.sigmoid().detach()
    '''

    explainer = GNNExplainer(model, num_hops=2, log = False)
    feat_mask, edge_weights = explainer.explain_graph(g, x)
    return edge_weights.cpu().numpy()

def explain_pgmexplainer(model, task_type, g, x, target, include_edges=None):
    #Get pred_threshold by predict of model (Actually not use in Explainer)
    pred = model.forward(g, x).cpu()
    soft_pred = np.array(pred[0].data)
    pred_threshold = 0.1*np.max(soft_pred)

    #Implement Graph_Explainer with 
    e = pe.Graph_Explainer(model, g,
                            perturb_feature_list = [0],
                            perturb_mode = "mean",
                            perturb_indicator = "diff")
    pgm_nodes, p_values, candidates = e.explain(num_samples = 1000, percentage = 10, 
                            top_node = 12, p_threshold = 0.05, pred_threshold = pred_threshold)
    explanation = zip(pgm_nodes,p_values)

    #Importance of node = 1 - p-value, convert node importance to edge importance
    node_attr = np.zeros(x.shape[0])
    for node, p_value in explanation:
        node_attr[node] = 1-p_value
    edge_mask = node_attr_to_edge(g, node_attr)


    return edge_mask

def explain_subgraphx(model, node_idx, x, edge_index, target, include_edges=None):
    if int(model.fc.out_features) == 6:
        max_nodes = 5
    else:
        max_nodes = 10

    explainer = SubgraphX(model, num_classes = int(model.fc.out_features), device = device, explain_graph = False, reward_method='nc_mc_l_shapley', local_radius =len(model.convs),
    num_hops = len(model.convs))

    _, explanation_results, related_preds = explainer(x, edge_index, node_idx=node_idx, max_nodes=max_nodes)

    explanation_results = explanation_results[target]

    explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)

    tree_node_x = find_closest_node_result(explanation_results, max_nodes = max_nodes)

    #tree_node_x.coalition#连通子图对应的结点列表
    #tree_node_x.ori_graph#原始图 nx结构
    important_edge = list(tree_node_x.ori_graph.subgraph(tree_node_x.coalition).edges)
    edge_mask = edge_index.new_zeros(edge_index.shape[1])
    for i in range(edge_index.shape[1]):
        if tuple(edge_index[:, i]) in important_edge:
            edge_mask[i] = 1
    return edge_mask.cpu().numpy()