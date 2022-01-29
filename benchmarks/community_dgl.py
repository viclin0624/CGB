
import random
from collections import defaultdict

import mlflow
import networkx as nx
import numpy as np
import torch
#from torch_geometric.utils import k_hop_subgraph, from_networkx
import dgl
from utils import k_hop_subgraph_dgl

from tqdm import tqdm


from benchmark_dgl import Benchmark
#from explain_methods import explain_pgmexplainer


class Rewiring:
    """dummy class for protecting these attributes from batch collation or movement to CUDA"""
    pass


class Community(Benchmark):
    NUM_GRAPHS = 50
    TEST_RATIO = 0.1
    EXPLANATION_SAMPLE_PER_GRAPH = 1000

    def __init__(self, sample_count, num_layers, concat_features, conv_type):
        super().__init__(sample_count, num_layers, concat_features, conv_type)
        mlflow.log_param('EXPLANATION_SAMPLE_PER_GRAPH', self.EXPLANATION_SAMPLE_PER_GRAPH)

    def create_dataset(self):
        K = 10
        SZ = 100
        N = SZ * K
        P = 0.05
        Q = 0.007
        PERTURB_COUNT = 500
        REWIRE_COUNT = 50
        mlflow.log_param('P', P)
        mlflow.log_param('Q', Q)
        mlflow.log_param('N', N)
        mlflow.log_param('K', K)
        mlflow.log_param('PERTURB_COUNT', PERTURB_COUNT)
        mlflow.log_param('REWIRE_COUNT', REWIRE_COUNT)
        sizes = [SZ] * K
        #P connect in block, Q connect between blocks
        probs = np.ones((K, K)) * Q
        probs += np.eye(K) * (P - Q)
        g = nx.stochastic_block_model(sizes, probs, directed=True)

        colors = []
        for i in range(K):
            colors.extend([i] * SZ)
        labels = colors.copy() #colors is feature, labels is labels
        #random choice color for nodes
        for i in random.sample(list(range(N)), PERTURB_COUNT):
            choices = list(range(10))
            # choices.remove(colors[i])
            colors[i] = random.choice(choices)
            # colors[i]=0

        features = np.zeros((len(g.nodes()), 10))
        for i, c in enumerate(colors):
            features[i, c] = 1

        data = dgl.from_networkx(g)
        data.ndata['x'] = torch.tensor(features, dtype=torch.float)
        data.ndata['y'] = torch.tensor(labels)

        edge_to_id = {}
        id_to_edge = {}
        bad_edges = defaultdict(list)
        for eid in range(data.num_edges()):
            u = data.edges()[0][eid]
            v = data.edges()[1][eid]
            u, v = u.item(), v.item()
            edge_to_id[(u, v)] = eid
            id_to_edge[eid] = (u, v)
            if labels[u] != labels[v]:
                bad_edges[labels[v]].append((u, v))# u v 相连但不是同一类，将被指的作为key保存下来

        node_edits = {}
        edit_type = {}
        for i in g.nodes():
            node_edits[i] = []
            for edit_type in ['good', 'bad']:
                edits = {}
                nodes_with_same_label = [x for x in g.nodes() if labels[x] == labels[i]]
                if edit_type == 'good':#将坏的链接改为好的
                    rewires = random.sample(bad_edges[labels[i]], REWIRE_COUNT)
                    new_edges = set()
                    for u, v in rewires:
                        assert labels[v] == labels[i]
                        u2 = random.choice(nodes_with_same_label) #把坏的链接的u改为正确类的u
                        while u2 == v or (u2, v) in edge_to_id or (u2, v) in new_edges:
                            u2 = random.choice(nodes_with_same_label)

                        edits[edge_to_id[(u, v)]] = (u2, v)#记录这个eid变成了什么边
                        new_edges.add((u2, v))#新边的集合
                else:#将好的链接改为坏的
                    new_edges = set()
                    while len(edits) < REWIRE_COUNT:
                        eid = random.randint(1, data.num_edges()) - 1
                        u, v = id_to_edge[eid]
                        if labels[u] == labels[i] or labels[v] == labels[i]:#原本两端都不在所属这一类里
                            continue
                        v2 = random.choice(nodes_with_same_label)#把指向的一端改到这一类
                        while (u, v2) in edge_to_id or (u, v2) in new_edges:#如果已经有了 1.已经存在 2.在之前添加的坏链接里已经添加过了，就要重新随机
                            v2 = random.choice(nodes_with_same_label)
                        edits[edge_to_id[(u, v)]] = (u, v2)
                        new_edges.add((u, v2))
                assert len(edits) == REWIRE_COUNT
                node_edits[i].append((edit_type, edits))#存两个元组

        rewiring = Rewiring()
        rewiring.id_to_edge = id_to_edge
        rewiring.edge_to_id = edge_to_id
        rewiring.node_edits = node_edits
        data.rewiring = rewiring
        data.num_classes = len(set(labels))
        data.num_node_features = data.ndata['x'].shape[1]

        return data

    def subsample_nodes(self, explain_function, nodes):
        #if explain_function.explain_function != explain_pgmexplainer:
            #return random.sample(nodes, self.EXPLANATION_SAMPLE_PER_GRAPH)
        return random.sample(nodes, self.PGMEXPLAINER_SUBSAMPLE_PER_GRAPH)

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name):
        accs = []
        for dss in test_dataset:
            bads = 0
            before_afters = []
            depth_limit = len(model.convs)
            tests = 0
            nodes_to_test = list(range(1000))
            nodes_to_test = self.subsample_nodes(explain_function, nodes_to_test)
            pbar = tqdm(nodes_to_test)
            dss = dss.to(self.device)
            model_cache = model(dss, dss.ndata['x'])
            #edge_index_rewired = dss.edge_index.clone().to(self.device)
            edge_index_rewired = dss.edges()
            rewired_graph = dgl.graph(edge_index_rewired)

            rewire_mask = torch.zeros(dss.num_edges(), dtype=bool)
            mask_edge_count = []
            for node_idx in pbar:
                prob, label = model_cache[[node_idx]].softmax(dim=1).max(dim=1)#开始实际执行rewire
                for edit_type, edits in dss.rewiring.node_edits[node_idx]:
                    for eid, (u, v) in edits.items():
                        rewired_graph.edges()[0][eid] = u
                        rewired_graph.edges()[1][eid] = v
                        rewire_mask[eid] = True

                    prob_rewired, label_rewired = model(rewired_graph, dss.ndata['x'])[[node_idx]].softmax(dim=1).max(dim=1)
                    target = dss.ndata['y'][node_idx].item()
                    should_test_explanation = False
                    if edit_type == 'good' and label_rewired.item() == target and prob_rewired.item() > prob.item():
                        should_test_explanation = True
                    if edit_type == 'bad' and prob_rewired.item() < prob.item():
                        should_test_explanation = True
                    # print(edit_type, 'should_test_explanation', should_test_explanation)
                    if should_test_explanation:
                        final_mask = (k_hop_subgraph_dgl(dss,node_idx, depth_limit)[4] &
                                      k_hop_subgraph_dgl(rewired_graph,node_idx, depth_limit)[4])

                        final_mask = final_mask.cpu() & rewire_mask
                        mask_edge_count.append(final_mask.sum().item())
                        attribution = explain_function(model, node_idx, dss.ndata['x'], dss.edges(), target, final_mask)[
                            final_mask]
                        attribution_rewired = \
                            explain_function(model, node_idx, dss.ndata['x'], edge_index_rewired, target, final_mask)[final_mask]

                        before_afters.append((attribution.mean(), attribution_rewired.mean()))
                        if edit_type == 'good' and attribution.mean() > attribution_rewired.mean():
                            bads += 1
                        if edit_type == 'bad' and attribution.mean() < attribution_rewired.mean():
                            bads += 1

                        tests += 1
                        pbar.set_postfix(bads=bads / (tests), tests=tests)
                    # revert to original
                    for eid in edits:
                        edge_index_rewired[0][eid] = torch.tensor(dss.rewiring.id_to_edge[eid], dtype=int)[0]
                        edge_index_rewired[1][eid] = torch.tensor(dss.rewiring.id_to_edge[eid], dtype=int)[1]
                        rewire_mask[eid] = False
            mlflow.log_metric('mask_edge_count', np.mean(mask_edge_count))
            mlflow.log_metric('tested_nodes_per_graph', tests)
            accs.append((1 - bads / tests))
        return accs
if __name__ == '__main__':
    A = Community(10, 4, True, 'GraphConvWL')
    A.run()