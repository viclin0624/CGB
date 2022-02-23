
import sys
sys.path.append('..')
from model.models_dgl import FixedNet2
from build_graph import build_graph
import dgl
import torch
import numpy as np
from collections import Counter
import typer
from tqdm import tqdm
def generate_single_sample(label, perturb_type, nodes_num = 25, m = 1, perturb_dic = {}, 
seed = None, no_attach_init_nodes=False):
    '''
    return a networkx instance
    '''
    basis_type = "ba"
    which_type = label
    if which_type == 0:
        if perturb_type == 0:
            G, role_id, plug_id = build_graph(nodes_num, basis_type, [], start = 0, m = m, seed = seed, no_attach_init_nodes=no_attach_init_nodes)
        else:
            G, role_id, plug_id = build_graph(nodes_num - perturb_type, basis_type, [[perturb_dic[perturb_type]]], start = 0, m = m, seed = seed, no_attach_init_nodes=no_attach_init_nodes)
    else:
        list_shapes = [["house"]] * (which_type - 1) + [["five_cycle"]] * (3 - which_type)
        if perturb_type != 0:
            list_shapes = list_shapes + [[perturb_dic[perturb_type]]]
        G, role_id, plug_id = build_graph(nodes_num-10-perturb_type, basis_type, list_shapes, start = 0, m = m, seed = seed, no_attach_init_nodes=no_attach_init_nodes)
    return G

def run(graphs_num: int = typer.Argument(..., help = 'Graph nums'), 
    label: int = typer.Argument(..., help = 'Not class 0'), 
    nodes_num: int = typer.Argument(..., help = 'Nodes num'), 
    device: str = typer.Argument(...,help = 'cuda:x'),
    grid_num: int = typer.Argument(100, help = 'random choice from which granularity. Example: 100 means step is 0.01')):
    GRAPHS_NUM = graphs_num
    LABEL = label
    NODES_NUM = nodes_num
    device = torch.device(device)
    fixed_model = FixedNet2(1,4,2,False,'GraphConvWL')
    fixed_model.set_paramerters()
    fixed_model.to(device)
    result = []
    for _ in tqdm(range(GRAPHS_NUM)):
        G = generate_single_sample(LABEL, 0, nodes_num = NODES_NUM, m = 6, perturb_dic = {4:'square_diagonal'}, no_attach_init_nodes = True)
        g = dgl.from_networkx(G)
        g = g.to(device)
        x = torch.ones((25,1)).to(device)
        if LABEL!=0:
            motif_mask = [False for _ in range(g.num_edges())]
            correct_ids = list(range(NODES_NUM-10, NODES_NUM))
            for i in range(g.num_edges()):
                u = g.edges()[0][i].item()
                v = g.edges()[1][i].item()
                if u in correct_ids or v in correct_ids:
                    motif_mask[i] = True
        else:
            print('Label should not be 0.')
            break
        input_mask = torch.tensor(np.random.choice(np.arange(0,1,grid_num), size = g.num_edges(), replace=True), dtype = torch.float32).to(device)
        input_mask[motif_mask] = 1.0
        result.append(np.argmax(fixed_model(g,x,input_mask).cpu().detach().numpy()))
    print(Counter(result))

if __name__ == '__main__':
    typer.run(run)

