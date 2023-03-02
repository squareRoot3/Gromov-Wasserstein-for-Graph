import time
import utils.spectralGW as sgw
import json
import utils.GromovWassersteinGraphToolkit as GwGt
from utils.GromovWassersteinFramework import *
from utils.GromovWassersteinGraphToolkit import *
import pickle
import warnings
from networkx.algorithms.community.quality import performance, coverage, modularity
from sklearn import metrics
from BAPG import *
warnings.filterwarnings("ignore")


def graph_partition_gd2(cost_s, p_s, p_t,idx2node, ot_hyperpara, trans0=None):
    """
    ** May 19, 2020: Gradient descent version of graph_partition
    
    
    Achieve a single graph partition via calculating Gromov-Wasserstein discrepancy
    between the target graph and proposed one

    Args:
        cost_s: (n_s, n_s) adjacency matrix of source graph
        p_s: (n_s, 1) the distribution of source nodes
        p_t: (n_t, 1) the distribution of target nodes
        idx2node: a dictionary {key = idx of row in cost, value = name of node}
        ot_hyperpara: a dictionary of hyperparameters

    Returns:
        sub_costs: a dictionary {key: cluster idx,
                                 value: sub cost matrices}
        sub_probs: a dictionary {key: cluster idx,
                                 value: sub distribution of nodes}
        sub_idx2nodes: a dictionary {key: cluster idx,
                                     value: a dictionary mapping indices to nodes' names
        trans: (n_s, n_t) the optimal transport
    """
    cost_t = np.diag(p_t[:, 0])
    cost_s = np.asarray(cost_s)
    # cost_t = 1 / (1 + cost_t)
    trans, log = gwa.gromov_wasserstein_asym_fixed_initialization(cost_s, cost_t, p_s.flatten(), p_t.flatten(), trans0)
    d_gw = log['gw_dist']
    sub_costs, sub_probs, sub_idx2nodes = node_cluster_assignment(cost_s, trans, p_s, p_t, idx2node)
    return sub_costs, sub_probs, sub_idx2nodes, trans, d_gw


# dictionaries for holding results
scores = {}
runtimes = {}
avetimes = {}

# load data
num_nodes = 1991
num_partitions = 12

with open('data/India_database.p', 'rb') as f:
    database = pickle.load(f)
G = nx.Graph()
nG = nx.Graph()
for i in range(num_nodes):
    G.add_node(i)
    nG.add_node(i)
for edge in database['edges']:
    G.add_edge(edge[0], edge[1])
    nG.add_edge(edge[0], edge[1])

start_edges = nx.number_of_edges(G)


# Load precomputed noisy version
save_name = "data/village_noise.txt"

with open(save_name, "rb") as fp:
    nG = pickle.load(fp)

database['labels'] = database['label']
print('---Data files loaded. Computing...\n')



def process_sgwl_village(cost,database,num_nodes,num_partitions,verbose=False):
    p_s = np.zeros((num_nodes, 1))
    p_s[:, 0] = np.sum(cost, axis=1) ** 0.001
    p_s /= np.sum(p_s)
    p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)
#     p_s = database['prob'] + 5e-1
#     p_s /= np.sum(p_s)
#     p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)

    ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
               'ot_method': 'proximal',
               'beta': 5e-5,
               'outer_iteration': 200,
               # outer, inner iteration, error bound of optimal transport
               'iter_bound': 1e-30,
               'inner_iteration': 1,
               'sk_bound': 1e-30,
               'node_prior': 0,
               'max_iter': 200,  # iteration and error bound for calcuating barycenter
               'cost_bound': 1e-16,
               'update_p': False,  # optional updates of source distribution
               'lr': 0,
               'alpha': 0}

    time_s = time.time()
    sub_costs, sub_probs, sub_idx2nodes, trans, d_gw = graph_partition_gd2(cost,
                                                                      p_s,
                                                                      p_t,
                                                                      database['idx2node'],
                                                                      ot_dict)
    est_idx = np.argmax(trans, axis=1)

    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx,average_method='max')

    if verbose:
        print('---Mutual information score = {:3.3f}'.format(mutual_info))

    return mutual_info, d_gw, trans



###########################################################
# Method: SpecGWL
###########################################################

# Note that the GWL pipeline above takes the true number of clusters as input.
# We now show how this number is estimated in the SpecGWL pipeline for
# a bona fide unsupervised partitioning method.

# def t_selection_pipeline_undirected_village(G,ts,fraction_t_to_keep=0.25):
#
#     mis = []
#     coups = []
#     d_gws = []
#     rt = []
#
#     for t in ts:
#         start = time.time()
#         cost = sgw.undirected_normalized_heat_kernel(G,t)
#         mutual_info, d_gw, coup = process_sgwl_village(cost,database,num_nodes,num_partitions)
#         mis.append(mutual_info)
#         coups.append(coup)
#         d_gws.append(d_gw)
#         end = time.time()
#         rt.append(end-start)
#
#     print('Couplings Computed')
#
#     coverages = []
#
#     for j in range(len(ts)):
#         coup = coups[j]
#         partition = get_partition(coup)
#         coverages.append(coverage(G,partition))
#
#     num_to_keep = int(np.round(fraction_t_to_keep*len(ts)))
#
#     good_t_max = ts[np.argsort(coverages)][-num_to_keep:]
#     good_t_grad = ts[np.argsort(np.abs(np.gradient(coverages)))][:num_to_keep]
#
#     return mis, coups, d_gws, good_t_max, good_t_grad, rt
#
#
# # Keeping t fixed, do a grid search to estimate the number of clusters
# num_clusts = list(range(5,45))
# t = 20
#
# cost = sgw.undirected_normalized_heat_kernel(G,t)
#
# d_gws = []
# mis = []
# coverages = []
# modularities = []
#
# for j in num_clusts:
#     mutual_info, d_gw, coup = process_sgwl_village(cost,database,num_nodes,j)
#     partition = get_partition(coup)
#     mis.append(mutual_info)
#     d_gws.append(d_gw)
#     coverages.append(coverage(G,partition))
#     modularities.append(modularity(G,partition))
#
# # Estimate number of clusters
# estimated_clusters_raw_sym = num_clusts[np.argmax(modularities)]
# print('Number of Clusters:',estimated_clusters_raw_sym)
#
# # Now perform modularity/coverage maximizing pipeline
# ts = np.linspace(5,50,20)
# mis, coups, d_gws, good_t_max, good_t_grad, rt = t_selection_pipeline_undirected_village(G,ts,estimated_clusters_raw_sym)
#
# coverages = []
#
# for j in range(len(ts)):
#     coup = coups[j]
#     partition = get_partition(coup)
#     coverages.append(coverage(G,partition))
#
# village_raw_sym_ami = mis[np.argmax(coverages)]
# print('AMI for VILLAGE, Raw, Sym:',village_raw_sym_ami)
# print('Occurs at t-value:',ts[np.argmax(coverages)])
# scores['specgwl-symmetric-raw'] = village_raw_sym_ami
# runtimes['specgwl-symmetric-raw'] = rt[np.argmax(coverages)]
#
# ## Repeat for noisy data
# num_clusts = list(range(5,20))
# t = 20
#
# cost = sgw.undirected_normalized_heat_kernel(nG,t)
#
# d_gws = []
# mis = []
# coverages = []
# modularities = []
#
# for j in num_clusts:
#     mutual_info, d_gw, coup = process_sgwl_village(cost,database,num_nodes,j)
#     partition = get_partition(coup)
#     mis.append(mutual_info)
#     d_gws.append(d_gw)
#     coverages.append(coverage(nG,partition))
#     modularities.append(modularity(nG,partition))
#
# estimated_clusters_noisy_sym = num_clusts[np.argmax(modularities)]
# print('Number of Clusters:',estimated_clusters_noisy_sym)
#
# ts = np.linspace(20,50,20)
# mis, coups, d_gws, good_t_max, good_t_grad, rt = t_selection_pipeline_undirected_village(nG,ts,estimated_clusters_noisy_sym)
#
# coverages = []
#
# for j in range(len(ts)):
#     coup = coups[j]
#     partition = get_partition(coup)
#     coverages.append(coverage(nG,partition))
#
# village_noisy_sym_ami = mis[np.argmax(coverages)]
# print('AMI for VILLAGE, Noisy, Sym:',village_noisy_sym_ami)
# print('Occurs at t-value:',ts[np.argmax(coverages)])
# scores['specgwl-symmetric-noisy'] = village_noisy_sym_ami
# runtimes['specgwl-symmetric-noisy'] = rt[np.argmax(coverages)]
#
# print('Mutual information scores')
# print(json.dumps(scores,indent=1))
# print('Runtimes')
# print(json.dumps(runtimes,indent=1))


num_partitions_clean = 9
t = 7.3684210526315797
num_partitions_noise = 9
t_noise = 10

####################################################################################################################
# Method: BAPG
####################################################################################################################

A = sgw.undirected_normalized_heat_kernel(G,t)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A, axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions_clean)

for rho in [0.001, 0.0005, 0.0001]:
    start = time.time()
    A = torch.tensor(A).float().cuda()
    p_s = torch.tensor(p_s).float().cuda()
    p_t = torch.tensor(p_t).float().cuda()
    coup_bap, obj = BAPG_torch(A, B=torch.eye(num_partitions_clean).cuda(),
                            a=p_s, b=p_t, epoch=2000, rho=1e-1, min_rho=rho, scaling=1.02) # start from a larger rho is more stable
    end = time.time()

    est_idx = np.argmax(coup_bap.cpu().numpy(), 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('BAPG Raw Heat rho: {}, AMI: {:.4f}'.format(rho, mutual_info))

A = sgw.undirected_normalized_heat_kernel(nG, t_noise)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A, axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions_noise)

A = torch.tensor(A).float().cuda()
p_s = torch.tensor(p_s).float().cuda()
p_t = torch.tensor(p_t).float().cuda()

for rho in [0.001, 0.0005, 0.0001]:
    A = torch.tensor(A).float().cuda()
    p_s = torch.tensor(p_s).float().cuda()
    p_t = torch.tensor(p_t).float().cuda()
    coup_bap, obj = BAPG_torch(A, B=torch.eye(num_partitions_noise).cuda(),
                               a=p_s, b=p_t, epoch=2000, rho=1e-1, min_rho=rho, scaling=1.02)
    est_idx = np.argmax(coup_bap.cpu().numpy(), 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('BAPG Noisy Heat rho: {}, AMI: {:.4f}'.format(rho, mutual_info))


############################### BAPG Clean Adj
A = nx.to_numpy_array(G)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A+np.eye(len(A)), axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)

A = torch.tensor(A).float().cuda()
p_s = torch.tensor(p_s).float().cuda()
p_t = torch.tensor(p_t).float().cuda()
for rho in [0.1, 0.05, 0.01]:
    coup_bap, obj = BAPG_torch(A, B=torch.eye(num_partitions).cuda(),
                               a=p_s, b=p_t, epoch=2000, rho=rho, min_rho=rho)
    est_idx = np.argmax(coup_bap.cpu().numpy(), 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('BAPG Raw Adj rho: {}, AMI: {:.4f}'.format(rho, mutual_info))

############################### BAPG Noise Adj
A = nx.to_numpy_array(nG)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A+np.eye(len(A)), axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)

for rho in [0.1, 0.05, 0.01]:
    A = torch.tensor(A).float().cuda()
    p_s = torch.tensor(p_s).float().cuda()
    p_t = torch.tensor(p_t).float().cuda()
    coup_bap, obj = BAPG_torch(A, B=torch.eye(num_partitions).cuda(),
                               a=p_s, b=p_t, epoch=2000, rho=rho, min_rho=rho)
    est_idx = np.argmax(coup_bap.cpu().numpy(), 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('BAPG Noisy Adj rho: {}, AMI: {:.4f}'.format(rho, mutual_info))


####################################################################################################################
# Method: eBPG
####################################################################################################################

A = sgw.undirected_normalized_heat_kernel(G,t)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A, axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions_clean)

for epsilon in [0.001, 0.0005, 0.0002, 0.0001]:
    coup_adj = ot.gromov.entropic_gromov_wasserstein(A, np.eye(num_partitions_clean), p_s.squeeze(-1), p_t.squeeze(-1),
                                                        loss_fun='kl_loss', epsilon=epsilon,
                                                        verbose=False, log=False, max_iter=1000)
    est_idx = np.argmax(coup_adj, 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('eBPG Raw Heat epsilon: {}, AMI: {:.4f}'.format(epsilon, mutual_info))


A = sgw.undirected_normalized_heat_kernel(nG,t)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A, axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions_noise)

for epsilon in [0.001, 0.0005, 0.0002, 0.0001]:
    coup_adj = ot.gromov.entropic_gromov_wasserstein(A.astype(np.float64), np.eye(num_partitions_noise).astype(np.float64),
                                                     p_s.squeeze(-1).astype(np.float64), p_t.squeeze(-1).astype(np.float64),
                                                        loss_fun='kl_loss', epsilon=epsilon,
                                                        verbose=False, log=False, max_iter=1000)
    est_idx = np.argmax(coup_adj, 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('eBPG Noisy Heat epsilon: {}, AMI: {:.4f}'.format(epsilon, mutual_info))

A = nx.to_numpy_array(G)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A+np.eye(len(A)), axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)

for epsilon in [0.1,0.01,0.001]:
    coup_adj = ot.gromov.entropic_gromov_wasserstein(A, np.eye(num_partitions), p_s.squeeze(-1), p_t.squeeze(-1),
                                                        loss_fun='square_loss', epsilon=epsilon,
                                                        verbose=False, log=False, max_iter=100)
    est_idx = np.argmax(coup_adj, 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('eBPG Raw Adj epsilon: {}, AMI: {:.4f}'.format(epsilon, mutual_info))

A = nx.to_numpy_array(nG)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A+np.eye(len(A)), axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)

for epsilon in [0.1,0.01,0.001]:
    coup_adj = ot.gromov.entropic_gromov_wasserstein(A, np.eye(num_partitions), p_s.squeeze(-1), p_t.squeeze(-1),
                                                        loss_fun='square_loss', epsilon=epsilon,
                                                        verbose=False, log=False, max_iter=100)
    est_idx = np.argmax(coup_adj, 1)
    mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
    print('eBPG Noisy Adj epsilon: {}, AMI: {:.4f}'.format(epsilon, mutual_info))


####################################################################################################################
# Method: BPG
####################################################################################################################
A = sgw.undirected_normalized_heat_kernel(G,t)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A, axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions_clean)
ot_hyperpara_adj = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                    'ot_method': 'proximal',
                    'beta': 0.001,  # 2
                    'outer_iteration': 500,
                    'iter_bound': 1e-10,
                    'inner_iteration': 500,  # origin: 1, BPG:500
                    'sk_bound': 1e-5,  # origin: 1e-30, BPG:1e-5
                    'node_prior': 0,
                    'max_iter': 200,  # iteration and error bound for calcuating barycenter
                    'cost_bound': 1e-16,
                    'update_p': False,  # optional updates of source distribution
                    'lr': 0,
                    'alpha': 0}
start = time.time()
coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(A, np.eye(num_partitions_clean), p_s, p_t, ot_hyperpara_adj)
end = time.time()
est_idx = np.argmax(coup_adj, 1)
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
print('BPG Raw Heat time: {:.4f}, AMI: {:.4f}'.format(end-start, mutual_info))

A = sgw.undirected_normalized_heat_kernel(nG,t_noise)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A, axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions_noise)
start = time.time()
coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(A, np.eye(num_partitions_noise), p_s, p_t, ot_hyperpara_adj)
end = time.time()
est_idx = np.argmax(coup_adj, 1)
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
print('BPG Noisy Heat time: {:.4f}, AMI: {:.4f}'.format(end-start, mutual_info))


A = nx.to_numpy_array(G)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A+np.eye(len(A)), axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)
start = time.time()
coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(A, np.eye(num_partitions), p_s, p_t, ot_hyperpara_adj)
end = time.time()
est_idx = np.argmax(coup_adj, 1)
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
print('BPG Raw Adj time: {:.4f}, AMI: {:.4f}'.format(end-start, mutual_info))


A = nx.to_numpy_array(nG)
p_s = np.zeros((len(A), 1))
p_s[:, 0] = np.sum(A+np.eye(len(A)), axis=1) ** .001
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)
start = time.time()
coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(A, np.eye(num_partitions), p_s, p_t, ot_hyperpara_adj)
end = time.time()
est_idx = np.argmax(coup_adj, 1)
mutual_info = metrics.adjusted_mutual_info_score(database['labels'], est_idx, average_method='max')
print('BPG Noisy Adj time: {:.4f}, AMI: {:.4f}'.format(end-start, mutual_info))