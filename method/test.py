import sys
sys.path.append('..')
import dgl
import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax

from method.pgm_explainer_example import FixedNet2
from method.pgm_explainer_example import Graph_Explainer
from method.pgm_explainer_example import node_attr_to_edge


model = FixedNet2(1,2,2,8)
g = dgl.graph((torch.tensor([i for i in range(50)]),torch.tensor([i for i in range(1,51)])))
g = dgl.add_self_loop(g)
x = torch.ones((g.num_nodes(),1))
pred = model.forward(g, x)
soft_pred = np.asarray(softmax(np.asarray(pred[0].data)))
pred_threshold = 0.1*np.max(soft_pred)
e = Graph_Explainer(model, g,
                        perturb_feature_list = [0],
                        perturb_mode = "mean",
                        perturb_indicator = "diff")
pgm_nodes, p_values, candidates = e.explain(num_samples = 1000, percentage = 10, 
                        top_node = 5, p_threshold = 0.05, pred_threshold = pred_threshold)
label = np.argmax(soft_pred)
pgm_nodes_filter = [i for i in pgm_nodes if p_values[i] < 0.02]
explanation = zip(pgm_nodes,p_values)
node_attr = np.zeros(x.shape[0])
for node, p_value in explanation:
    node_attr[node] = 1 - p_value
edge_mask = node_attr_to_edge(g, node_attr)
print(edge_mask)