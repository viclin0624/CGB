import argparse
import os
import xdrlib
import dgl

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl import load_graphs
from models import dummy_gnn_model
from NodeExplainerModule import NodeExplainerModule
from utils_graph import extract_subgraph, visualize_sub_graph


def main(model, task_type, g, x, target):
    # load graph, feat, and label
    labels = target
    feats = x
    dummy_model = model
    num_classes = max(labels).item() + 1
    feat_dim = feats.shape[1]
    args.lr = 0.01
    args.wd = 0
    args.epochs = 200

    # create an explainer
    explainer = NodeExplainerModule(model=model,
                                    num_edges=g.number_of_edges(),
                                    node_feat_dim=feat_dim)

    # define optimizer
    optim = th.optim.Adam(explainer.parameters(), lr=args.lr, weight_decay=args.wd)

    # train the explainer for the given node
    dummy_model.eval()
    model_logits = dummy_model(g, x)
    model_predict = F.one_hot(th.argmax(model_logits, dim=-1), num_classes)

    for epoch in range(args.epochs):
        explainer.train()
        exp_logits = explainer(g, x)
        loss = explainer._loss(exp_logits, model_predict)

        optim.zero_grad()
        loss.backward()
        optim.step()

    # visualize the importance of edges
    edge_weights = explainer.edge_mask.sigmoid().detach()
    return edge_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of GNN explainer in DGL')
    parser.add_argument('--dataset', type=str, default='syn1',
                        help='The dataset to be explained.')
    parser.add_argument('--target_class', type=int, default='1',
                        help='The class to be explained. In the synthetic 1 dataset, Valid option is from 0 to 4'
                             'Will choose the first node in this class to explain')
    parser.add_argument('--hop', type=int, default='2',
                        help='The hop number of the computation sub-graph. For syn1 and syn2, k=2. For syn3, syn4, and syn5, k=4.')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay.')
    args = parser.parse_args()
    print(args)

    main(args)