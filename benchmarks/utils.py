#%%
import torch
import dgl
from collections.abc import Mapping
from dgl import DGLError
import tensor_backend as F
import pandas as pd
def prepare_tensor(g, data, name):
    """Convert the data to ID tensor and check its ID type and context.
    If the data is already in tensor type, raise error if its ID type
    and context does not match the graph's.
    Otherwise, convert it to tensor type of the graph's ID type and
    ctx and return.
    Parameters
    ----------
    g : DGLHeteroGraph
        Graph.
    data : int, iterable of int, tensor
        Data.
    name : str
        Name of the data.
    Returns
    -------
    Tensor
        Data in tensor object.
    """
    if torch.is_tensor(data):
        if data.dtype != g.idtype or data.device != g.device:
            raise DGLError('Expect argument "{}" to have data type {} and device '
                           'context {}. But got {} and {}.'.format(
                               name, g.idtype, g.device, data.dtype, data.device))
        ret = data
    else:
        data = torch.tensor(data)
        if (not (data.ndim > 0 and data.shape[0] == 0) and        # empty tensor
                data.dtype not in (torch.int32, torch.int64)):
            raise DGLError('Expect argument "{}" to have data type int32 or int64,'
                           ' but got {}.'.format(name, data.dtype))
        ret = F.copy_to(F.astype(data, g.idtype), g.device)

    if ret.ndim == 0:
        ret = torch.unsqueeze(ret, 0)
    if ret.ndim > 1:
        raise DGLError('Expect a 1-D tensor for argument "{}". But got {}.'.format(
            name, ret))
    return ret

def khop_out_subgraph(graph, nodes, k, *, relabel_nodes=True, store_ids=True):
    """Return the subgraph induced by k-hop out-neighborhood of the specified node(s).
    We can expand a set of nodes by including the successors of them. From a
    specified node set, a k-hop out subgraph is obtained by first repeating the node set
    expansion for k times and then creating a node induced subgraph. In addition to
    extracting the subgraph, DGL also copies the features of the extracted nodes and
    edges to the resulting graph. The copy is *lazy* and incurs data movement only
    when needed.
    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus the resulting graph has the same set of relations
    as the input one.
    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The starting node(s) to expand. The allowed formats are:
        * Int: ID of a single node.
        * Int Tensor: Each element is a node ID. The tensor must have the same device
          type and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.
        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the node IDs in the above formats.
    k : int
        The number of hops.
    relabel_nodes : bool, optional
        If True, it will remove the isolated nodes and relabel the rest nodes in the
        extracted subgraph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the extracted nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    Returns
    -------
    DGLGraph
        The subgraph.
    Tensor or dict[str, Tensor], optional
        The new IDs of the input :attr:`nodes` after node relabeling. This is returned
        only when :attr:`relabel_nodes` is True. It is in the same form as :attr:`nodes`.
    Notes
    -----
    When k is 1, the result subgraph is different from the one obtained by
    :func:`dgl.out_subgraph`. The 1-hop out subgraph also includes the edges
    among the neighborhood.
    Examples
    --------
    The following example uses PyTorch backend.
    >>> import dgl
    >>> import torch
    Extract a two-hop subgraph from a homogeneous graph.
    >>> g = dgl.graph(([0, 2, 0, 4, 2], [1, 1, 2, 3, 4]))
    >>> g.edata['w'] = torch.arange(10).view(5, 2)
    >>> sg, inverse_indices = dgl.khop_out_subgraph(g, 0, k=2)
    >>> sg
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'w': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([0, 0, 2, 2]), tensor([1, 2, 1, 3]))
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([0, 2, 1, 4])
    >>> sg.edata['w']  # also extract the features
    tensor([[0, 1],
            [4, 5],
            [2, 3],
            [8, 9]])
    >>> inverse_indices
    tensor([0])
    Extract a subgraph from a heterogeneous graph.
    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1], [1, 3])})
    >>> sg, inverse_indices = dgl.khop_out_subgraph(g, {'user': 0}, k=2)
    >>> sg
    Graph(num_nodes={'game': 2, 'user': 3},
          num_edges={('user', 'follows', 'user'): 2, ('user', 'plays', 'game'): 2},
          metagraph=[('user', 'user', 'follows'), ('user', 'game', 'plays')])
    >>> inverse_indices
    {'user': tensor([0])}
    See also
    --------
    khop_in_subgraph
    """
    if graph.is_block:
        raise DGLError('Extracting subgraph of a block graph is not allowed.')

    is_mapping = isinstance(nodes, Mapping)
    if not is_mapping:
        assert len(graph.ntypes) == 1, \
            'need a dict of node type and IDs for graph with multiple node types'
        nodes = {graph.ntypes[0]: nodes}

    for nty, nty_nodes in nodes.items():
        nodes[nty] = prepare_tensor(graph, nty_nodes, 'nodes["{}"]'.format(nty))

    last_hop_nodes = nodes
    k_hop_nodes_ = [last_hop_nodes]
    place_holder = F.copy_to(F.tensor([], dtype=graph.idtype), graph.device)
    for _ in range(k):
        current_hop_nodes = {nty: [] for nty in graph.ntypes}
        for cetype in graph.canonical_etypes:
            srctype, _, dsttype = cetype
            _, out_nbrs = graph.out_edges(last_hop_nodes.get(
                srctype, place_holder), etype=cetype)
            current_hop_nodes[dsttype].append(out_nbrs)
        for nty in graph.ntypes:
            if len(current_hop_nodes[nty]) == 0:
                current_hop_nodes[nty] = place_holder
                continue
            current_hop_nodes[nty] = F.unique(F.cat(current_hop_nodes[nty], dim=0))
        k_hop_nodes_.append(current_hop_nodes)
        last_hop_nodes = current_hop_nodes

    k_hop_nodes = dict()
    inverse_indices = dict()
    for nty in graph.ntypes:
        k_hop_nodes[nty], inverse_indices[nty] = F.unique(F.cat([
            hop_nodes.get(nty, place_holder)
            for hop_nodes in k_hop_nodes_], dim=0), return_inverse=True)

    sub_g = dgl.node_subgraph(graph, k_hop_nodes, relabel_nodes=relabel_nodes, store_ids=store_ids)
    if relabel_nodes:
        if is_mapping:
            seed_inverse_indices = dict()
            for nty in nodes:
                seed_inverse_indices[nty] = F.slice_axis(
                    inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty]))
        else:
            seed_inverse_indices = F.slice_axis(
                inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty]))
        return sub_g, seed_inverse_indices
    else:
        return sub_g

def k_hop_subgraph_dgl(g, nodeid, k):
    if type(nodeid) != torch.tensor:
        if type(nodeid) == list:
            nid = torch.tensor(nodeid)
        elif type(nodeid) == int:
            nid = torch.tensor([nodeid])
        else:
            print('Don\'t support nodeid type: ',type(nodeid))
            return 
    else:
        nid = nodeid
    device = g.device
    nid = nid.to(device)
    nids = [nid]
    for _ in range(k):
        #nid = g.in_edges(nid)[0].unique()
        nid = g.out_edges(nid)[1].unique()
        nids.append(nid)
    nids = torch.cat(nids).unique()
    subgraph = g.subgraph(nids)
    edges_list = list(zip(*[g.edges()[0].cpu().numpy(), g.edges()[1].cpu().numpy()]))
    u = []
    v = []
    edges_bool_list  = torch.tensor([False]*g.num_edges())
    for index in subgraph.edata[dgl.EID]:
        u.append(edges_list[index][0])
        v.append(edges_list[index][1])
        edges_bool_list[index] = True
    edges_with_origin_node_id = (torch.tensor(u),torch.tensor(v))
    return subgraph, subgraph.edata[dgl.EID], subgraph.ndata[dgl.NID], edges_with_origin_node_id, edges_bool_list

def record_exp(result, exp_no, filepath = '/home/ubuntu/Maolin/eva_gnn/Exp.csv', mlist = [1,5], layerlist = [1,2,3,4]):
    if len(result[(mlist[0],layerlist[0])]) == 4:
        collist = ['exp','m','layers','train_acc','test_acc']
        for i in ['train_', 'test_']:
            for j in range(4):
                for k in ['_true','_total','_0','_1','_2','_3']:
                    collist.append(i+str(j)+k)
        D = pd.read_csv(filepath)
        i = exp_no
        for j in mlist:
            for k in layerlist:
                tmp = ['exp'+str(i),j,k]
                tmp.append(result[(j,k)][0])
                tmp.append(result[(j,k)][1])
                for l in range(4):
                    for m in range(6):
                        tmp.append(result[(j,k)][2][l][m])
                for l in range(4):
                    for m in range(6):
                        tmp.append(result[(j,k)][3][l][m])
                tmp = pd.DataFrame([tmp],columns=collist)
                D = D.append(tmp)
    elif len(result[(mlist[0],layerlist[0])]) == 6:
        filepath = '/home/ubuntu/Maolin/eva_gnn/Exp2.csv'
        collist = ['exp','m','layers','train_acc','test_acc','test2_acc']
        for i in ['train_', 'test_', 'test2_']:
            for j in range(4):
                for k in ['_true','_total','_0','_1','_2','_3']:
                    collist.append(i+str(j)+k)
        try:
            D = pd.read_csv(filepath)
        except:
            D = pd.DataFrame([],columns=collist)
        i = exp_no
        for j in mlist:
            for k in layerlist:
                tmp = ['exp'+str(i),j,k]
                tmp.append(result[(j,k)][0])
                tmp.append(result[(j,k)][1])
                tmp.append(result[(j,k)][2])
                for l in range(4):
                    for m in range(6):
                        tmp.append(result[(j,k)][3][l][m])
                for l in range(4):
                    for m in range(6):
                        tmp.append(result[(j,k)][4][l][m])
                for l in range(4):
                    for m in range(6):
                        tmp.append(result[(j,k)][5][l][m])
                tmp = pd.DataFrame([tmp],columns=collist)
                D = D.append(tmp)
    D.to_csv(filepath, index=False)