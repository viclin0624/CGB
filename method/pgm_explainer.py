import time
from pgmpy import device
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
#from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.estimators.CITests import chi_square
from scipy.stats import chi2_contingency
from scipy.stats import chisquare

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

'''def chi_square(x,y,data):
    chi, p_value, dof, expected = chi2_contingency(
            data.groupby([x, y]).size().unstack(y, fill_value=0), lambda_="pearson"
        )
    return chi, p_value'''
'''def chi_square(X, Y, Z, data, **kwargs):
    """
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.
    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    `P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as `P(X,Zs)*P(Y,Zs)/P(Zs).
    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set
    Y: int, string, hashable object
        A variable name contained in the data set, different from X
    Zs: list of variable names
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []
    Returns
    -------
    chi2: float
        The chi2 test statistic.
    p_value: float
        The p_value, i.e. the probability of observing the computed chi2
        statistic (or an even higher value), given the null hypothesis
        that X _|_ Y | Zs.
    sufficient_data: bool
        A flag that indicates if the sample size is considered sufficient.
        As in [4], require at least 5 samples per parameter (on average).
        That is, the size of the data set must be greater than
        `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
        (c() denotes the variable cardinality).
    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.2.2.3 (page 789)
    [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
    [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
    [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import ConstraintBasedEstimator
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> c = ConstraintBasedEstimator(data)
    >>> print(c.test_conditional_independence('A', 'C'))  # independent
    True
    >>> print(c.test_conditional_independence('A', 'B', 'D'))  # independent
    True
    >>> print(c.test_conditional_independence('A', 'B', ['D', 'E']))  # dependent
    False
    """

    if isinstance(Z, (frozenset, list, set, tuple)):
        Z = list(Z)
    else:
        Z = [Z]

    if "state_names" in kwargs.keys():
        state_names = kwargs["state_names"]
    else:
        state_names = {
            var_name: data.loc[:, var_name].unique() for var_name in data.columns
        }
    num_params = (
        (len(state_names[X]) - 1)
        * (len(state_names[Y]) - 1)
        * np.prod([len(state_names[z]) for z in Z])
    )
    sufficient_data = len(data) >= num_params * 5

    # compute actual frequency/state_count table:
    # = P(X,Y,Zs)
    XYZ_state_counts = pd.crosstab(
        index=data[X], columns=[data[Y]] + [data[z] for z in Z]
    )
    # reindex to add missing rows & columns (if some values don't appear in data)
    row_index = state_names[X]
    column_index = pd.MultiIndex.from_product(
        [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
    )
    if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
        XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
    XYZ_state_counts = XYZ_state_counts.reindex(
        index=row_index, columns=column_index
    ).fillna(0)

    # compute the expected frequency/state_count table if X _|_ Y | Zs:
    # = P(X|Zs)*P(Y|Zs)*P(Zs) = P(X,Zs)*P(Y,Zs)/P(Zs)
    if Z:
        XZ_state_counts = XYZ_state_counts.sum(axis=1, level=Z)  # marginalize out Y
        YZ_state_counts = XYZ_state_counts.sum().unstack(Z)  # marginalize out X
    else:
        XZ_state_counts = XYZ_state_counts.sum(axis=1)
        YZ_state_counts = XYZ_state_counts.sum()
    Z_state_counts = YZ_state_counts.sum()  # marginalize out both

    XYZ_expected = pd.DataFrame(
        index=XYZ_state_counts.index, columns=XYZ_state_counts.columns
    )
    for X_val in XYZ_expected.index:
        if Z:
            for Y_val in XYZ_expected.columns.levels[0]:
                XYZ_expected.loc[X_val, Y_val] = (
                    XZ_state_counts.loc[X_val]
                    * YZ_state_counts.loc[Y_val]
                    / Z_state_counts
                ).values
        else:
            for Y_val in XYZ_expected.columns:
                XYZ_expected.loc[X_val, Y_val] = (
                    XZ_state_counts.loc[X_val]
                    * YZ_state_counts.loc[Y_val]
                    / float(Z_state_counts)
                )

    observed = XYZ_state_counts.values.flatten()
    expected = XYZ_expected.fillna(0).values.flatten()
    # remove elements where the expected value is 0;
    # this also corrects the degrees of freedom for chisquare
    observed, expected = zip(
        *((o, e) for o, e in zip(observed, expected) if not e == 0)
    )

    chi2, significance_level = chisquare(observed, expected)

    return chi2, significance_level'''


class Graph_Explainer:
    def __init__(
        self,
        model,
        graph,
        num_layers = None,
        perturb_feature_list = None,
        perturb_mode = "mean", # mean, zero, max or uniform
        perturb_indicator = "diff", # diff or abs
        print_result = 1,
        snorm_n = None, 
        snorm_e = None
    ):
        self.model = model
        self.model.eval()
        self.graph = graph
        self.snorm_n = snorm_n
        self.snorm_e = snorm_e
        self.num_layers = num_layers
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        self.print_result = print_result
        self.device = next(self.model.parameters()).device
        self.X_feat = torch.ones((graph.num_nodes(),1),device = self.device)
        self.E_feat = torch.ones((graph.num_edges(),1),device = self.device)
    def perturb_features_on_node(self, feature_matrix, node_idx, random = 0):
        
        X_perturb = feature_matrix.detach().clone()
        perturb_array = X_perturb[node_idx].detach().clone()
        epsilon = 0.05*torch.max(self.X_feat, dim = 0)[0]
        seed = np.random.randint(2)
        
        if random == 1:
            if seed == 1:
                for i in range(perturb_array.shape[0]):
                    if i in self.perturb_feature_list:
                        if self.perturb_mode == "mean":
                            perturb_array[i] = torch.mean(feature_matrix[:,i])
                        elif self.perturb_mode == "zero":
                            perturb_array[i] = 0
                        elif self.perturb_mode == "max":
                            perturb_array[i] = torch.max(feature_matrix[:,i])
                        elif self.perturb_mode == "uniform":
                            perturb_array[i] = perturb_array[i] + np.random.uniform(low=-epsilon[i], high=epsilon[i])
                            if perturb_array[i] < 0:
                                perturb_array[i] = 0
                            elif perturb_array[i] > torch.max(self.X_feat, dim = 0)[0][i]:
                                perturb_array[i] = torch.max(self.X_feat, dim = 0)[0][i]

        
        X_perturb[node_idx] = perturb_array

        return X_perturb 
    
    def batch_perturb_features_on_node(self, num_samples, index_to_perturb,
                                            percentage, p_threshold, pred_threshold):
        X_torch = self.X_feat.detach().clone()
        E_torch = self.E_feat.detach().clone()
        pred_torch = self.model.forward(self.graph, X_torch, E_torch).cpu()
        soft_pred = np.asarray(softmax(np.asarray(pred_torch[0].data)))
        pred_label = np.argmax(soft_pred)
        num_nodes = self.X_feat.shape[0]
        Samples = [] 
        for iteration in range(num_samples):
            X_perturb = self.X_feat.detach().clone()
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
            
            X_perturb_torch =  X_perturb
            pred_perturb_torch = self.model.forward(self.graph, X_perturb_torch, E_torch).cpu()
            soft_pred_perturb = np.asarray(softmax(np.asarray(pred_perturb_torch[0].data)))
        
            pred_change = np.max(soft_pred) - soft_pred_perturb[pred_label]
            
            sample.append(pred_change)
            Samples.append(sample)
        
        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)
        
        top = int(num_samples/8)
        top_idx = np.argsort(Samples[:,num_nodes])[-top:] 
        for i in range(num_samples):
            if i in top_idx:
                Samples[i,num_nodes] = 1
            else:
                Samples[i,num_nodes] = 0
            
        return Samples
    
    def explain(self, num_samples = 10, percentage = 50, top_node = None, p_threshold = 0.05, pred_threshold = 0.1):

        num_nodes = self.X_feat.shape[0]
        if top_node == None:
            top_node = int(num_nodes/20)
        #start = time.time()
        #Round 1
        Samples = self.batch_perturb_features_on_node(int(num_samples/2), range(num_nodes),percentage, 
                                                            p_threshold, pred_threshold)         
        
        data = pd.DataFrame(Samples)

        #end = time.time()
        #print(end-start)
        p_values = []
        candidate_nodes = []
        
        #start = time.time()
        target = num_nodes # The entry for the graph classification data is at "num_nodes"
        for node in range(num_nodes):
            #print('----------')
            chi2, p = chi_square(node, target, [], data)
            #print(chi2, p)
            #chi2, p = chi_square2(node, target, [], data)
            #print(chi2, p)
            p_values.append(p)
        
        number_candidates = int(top_node*4)
        candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]
        #end = time.time()
        #print(end-start)
        #Round 2
        #start = time.time()
        Samples = self.batch_perturb_features_on_node(num_samples, candidate_nodes, percentage, 
                                                            p_threshold, pred_threshold)          
        data = pd.DataFrame(Samples)

        #end = time.time()
        #print(end-start)
        p_values = []
        dependent_nodes = []
        
        #start = time.time()
        target = num_nodes
        for node in range(num_nodes):
            #print('----------')
            chi2, p = chi_square(node, target, [], data)
            #print(chi2, p)
            #chi2, p = chi_square2(node, target, [], data)
            #print(chi2, p)
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)
        #end = time.time()
        #print(end-start)
        top_p = np.min((top_node,num_nodes-1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)
        
        return pgm_nodes, p_values, candidate_nodes
'''

Node classification

import numpy as np
import pandas as pd
import torch
from pgmpy.estimators.CITests import chi_square
from scipy.special import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Graph_Explainer:
    def __init__(
            self,
            model,
            g,
            X,
            num_layers,
            mode=0,
            print_result=1
    ):
        self.model = model
        self.model.eval()
        self.g = g
        self.X = X
        self.num_layers = num_layers
        self.mode = mode
        self.print_result = print_result

    def perturb_features_on_node(self, feature_matrix, node_idx, random=0, mode=0):
        # return a random perturbed feature matrix
        # random = 0 for nothing, 1 for random.
        # mode = 0 for random 0-1, 1 for scaling with original feature

        X_perturb = feature_matrix
        if mode == 0:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.random.randint(2, size=X_perturb[node_idx].shape[0])
            X_perturb[node_idx] = perturb_array
        elif mode == 1:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.multiply(X_perturb[node_idx],
                                            np.random.uniform(low=0.0, high=2.0, size=X_perturb[node_idx].shape[0]))
            X_perturb[node_idx] = perturb_array
        return X_perturb

    def explain(self, target, num_samples=100, top_node=None, p_threshold=0.05, pred_threshold=0.1):

        pred_torch = self.model(self.g, self.X).cpu()
        soft_pred = softmax(np.asarray(pred_torch.data))
        pred_node = np.asarray(pred_torch.data)
        label_node = np.argmax(pred_node)
        soft_pred_node = softmax(pred_node)

        Samples = []
        Pred_Samples = []

        for iteration in range(num_samples):

            X_perturb = self.X.cpu().detach().numpy()
            sample = []
            for node in self.g.nodes():
                seed = np.random.randint(2)
                if seed == 1:
                    latent = 1
                    X_perturb = self.perturb_features_on_node(X_perturb, node, random=seed)
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float).to(device)
            pred_perturb_torch = self.model(self.g, X_perturb_torch).cpu()
            soft_pred_perturb = softmax(np.asarray(pred_perturb_torch.data))

            sample_bool = []
            for node in self.g.nodes():
                if (soft_pred_perturb[node, target] + pred_threshold) < soft_pred[node, target]:
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            Samples.append(sample)
            Pred_Samples.append(sample_bool)

        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = Samples - Samples
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])])

        data = pd.DataFrame(Combine_Samples)
        data = data.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(self.g.nodes(), list(data.columns)))

        p_values = []
        for node in self.g.nodes():
            chi2, p = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [], data)
            p_values.append(p)

        pgm_stats = dict(zip(self.g.nodes(), p_values))

        return pgm_stats'''