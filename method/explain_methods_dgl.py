from typing import Union, Tuple, Any
from scipy.special import softmax

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from captum._utils.typing import TargetType
from captum.attr import Saliency, IntegratedGradients, LayerGradCam
from torch import Tensor
import dgl
from method.gnn_explainer import GNNExplainer
import method.pgm_explainer as pe
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_forward(input_mask, g, model, x):
    if input_mask.shape[0] != g.num_edges():
        out = []
        for i in range(int(input_mask.shape[0]/g.num_edges())):
            out.append(model(g, x[(i*g.num_nodes()):((i+1)*g.num_nodes())], input_mask[(i*g.num_edges()):((i+1)*g.num_edges())]))
        out = torch.cat(out, dim = 0)
    else:
        out = model(g, x, input_mask)
    return out




def node_attr_to_edge(g, node_mask):
    edge_mask = np.zeros(g.num_edges())
    edge_mask += node_mask[g.edges()[0].cpu().numpy()]
    edge_mask += node_mask[g.edges()[1].cpu().numpy()]
    return edge_mask



def explain_random(model, task_type, g, x, target):
    return np.random.uniform(size=g.num_edges())



def explain_sa(model, task_type, g, x, target):
    saliency = Saliency(model_forward)
    input_mask = torch.ones(g.num_edges()).requires_grad_(True).to(device)
    input_mask.retain_grad()
    attr = saliency.attribute(input_mask, target=int(target), additional_forward_args = (g,model,x), abs = True)
    attr = attr.detach().cpu().numpy()
    return attr


def explain_ig(model, task_type, g, x, target):
    ig = IntegratedGradients(model_forward)
    input_mask = torch.ones(g.num_edges()).requires_grad_(True).to(device)
    attr,delta = ig.attribute(input_mask, target=int(target), additional_forward_args = (g,model,x),return_convergence_delta=True, n_steps=500)
    attr = attr.detach().cpu().numpy()
    return attr



def explain_gnnexplainer(model, task_type, g, x, target):
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
                            perturb_mode = "uniform",
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