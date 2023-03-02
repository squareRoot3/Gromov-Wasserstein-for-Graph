import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import time
import ot
import scipy.sparse
import torch
from scipy import linalg
from scipy import sparse
import scipy.io
import utils.gromovWassersteinAveraging as gwa
import utils.spectralGW as sgw
from utils.GromovWassersteinFramework import *
import utils.GromovWassersteinGraphToolkit as GwGt
from BAPG import *
from collections import defaultdict
import pickle
import warnings
import argparse
warnings.filterwarnings("ignore")
seeds = [123]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='proteins', help='proteins / reddit / enzymes / synthetic')
parser.add_argument('--noise_level', type=float, default=0.)
args = parser.parse_args()
database = args.dataset
noise_level = args.noise_level

if database == 'proteins':
    print('------------------Node Matching on PROTIENS---------------')
    with open('data/PROTEINS/matching.pk', 'rb') as f:
        graphs, _ = pickle.load(f)

if database == 'reddit':
    print('------------------Node Matching on REDDIT---------------')
    with open('data/REDDIT-BINARY/matching.pk', 'rb') as f:
        graphs = pickle.load(f)[:500]

if database == 'enzymes':
    print('------------------Node Matching on ENZYMES---------------')
    with open('data/ENZYMES/matching.pk', 'rb') as f:
        graphs = pickle.load(f)

if database == 'synthetic':
    graphs, noise_graphs = [], []
    print('------------------Node Matching on Synthetic Database---------------')
    with open('data/Random/graph1.pk', 'rb') as f:
        graph_pairs = pickle.load(f)
        for num_node in [500, 1000, 1500, 2000, 2500]:
            for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                for G, G_noise in graph_pairs[(num_node, noise_level)]:
                    graphs.append(G)
                    noise_graphs.append(G_noise)

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if database != 'synthetic':
        if noise_level > 0:
            noise_graphs = []
            for G_src in graphs:
                G_dst = add_noisy_edges(G_src, noise_level)
                G_dst = add_noisy_nodes(G_dst, noise_level)
                noise_graphs.append(G_dst)
        else:
            noise_graphs = graphs

    total_num_graphs = len(graphs)
    print('total_num_graphs: ', total_num_graphs)

    results, times, error = defaultdict(list), defaultdict(list), defaultdict(list)

    for j in range(0, total_num_graphs):
        print('graph id: ', j)
        G = graphs[j]
        G_noise = noise_graphs[j]
        G_adj = nx.to_numpy_array(G).astype(np.float32)
        G_adj_noise = nx.to_numpy_array(G_noise).astype(np.float32)
        m, n = G_adj.shape[0], G_adj_noise.shape[0]
        p = np.ones([m,1]).astype(np.float32)/m
        q = np.ones([n,1]).astype(np.float32)/n
        Xinit = p @ q.T

        ######Our-BAPG-GPU###########################################################################################################################
        G_adj_gpu = torch.tensor(G_adj).cuda()
        G_adj_noise_gpu = torch.tensor(G_adj_noise).cuda()
        start = time.time()
        rho = 0.1
        coup_bap, obj_list_bap = BAPG_torch(A=G_adj_gpu, B=G_adj_noise_gpu, a=None, b=None, epoch=2000, eps=1e-5,
                                            rho=rho, min_rho=rho)
        end = time.time()
        times['BAPG'].append(end-start)
        results['BAPG'].append(node_correctness(coup_bap.cpu().numpy(), np.eye(m)))
        error['BAPG'].append(calculate_infeat(coup_bap.cpu().numpy(), p, q))
        ######Our-BAPG-CPU###########################################################################################################################
        # start = time.time()
        # rho = 0.1
        # coup_bap, obj_list_bap = BAPG_numpy(A=G_adj, B=G_adj_noise, a=p, b=q, X=Xinit, epoch=500, eps=1e-5,
        #                                     rho=rho)
        # end = time.time()
        # times['BAPGcpu'].append(end-start)
        # results['BAPGcpu'].append(node_correctness(coup_bap, np.eye(m)))

        ######FW###########################################################################################################################
        p = np.ones([m,1]).astype(np.float32)/m
        q = np.ones([n,1]).astype(np.float32)/n
        start = time.time()
        coup_adj, log = ot.gromov.gromov_wasserstein(G_adj, G_adj_noise, p.squeeze(), q.squeeze(),
                                                       loss_fun='kl_loss', log=True)
        end = time.time()
        times['FW'].append(end-start)
        results['FW'].append(node_correctness(coup_adj, np.eye(m)))
        error['FW'].append(calculate_infeat(coup_adj, p, q))
        #######BPG-S##############################################################################################################################
        idx2node_s, idx2node_t = {}, {}
        p_s, cost_s, idx2node_s = extract_graph_info(G, weights=None)
        p_s /= np.sum(p_s)
        p_t, cost_t, idx2node_t = extract_graph_info(G_noise, weights=None)
        p_t /= np.sum(p_t)

        start = time.time()
        ot_hyperpara_adj = {'loss_type': 'L2',
                            'ot_method': 'proximal',
                            'beta': 0.2,
                            'outer_iteration': 200,
                            'iter_bound': 1e-10,
                            'inner_iteration': 2,
                            'sk_bound': 1e-10,
                            'node_prior': 0,
                            'max_iter': 200,
                            'cost_bound': 1e-16,
                            'update_p': False,
                            'lr': 0,
                            'alpha': 0}
        coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(G_adj, G_adj_noise, p, q, ot_hyperpara_adj)
        end = time.time()
        times['BPGS'].append(end-start)
        results['BPGS'].append(node_correctness(coup_adj, np.eye(m)))
        error['BPGS'].append(calculate_infeat(coup_adj, p, q))
        #########ScalaGW##############################################################################################################################
        ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                   'ot_method': 'proximal',
                   'beta': 0.2,  #
                   'outer_iteration': 200,  # outer, inner iteration, error bound of optimal transport
                   'iter_bound': 1e-10,
                   'inner_iteration': 2,
                   'sk_bound': 1e-10,
                   'node_prior': 0,
                   'max_iter': 5,  # iteration and error bound for calcuating barycenter
                   'cost_bound': 1e-16,
                   'update_p': False,  # optional updates of source distribution
                   'lr': 0,
                   'alpha': 0}
        start = time.time()
        pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
            cost_s, cost_t, p_s, p_t, idx2node_s, idx2node_t, ot_dict,
            weights=None, predefine_barycenter=False, cluster_num=8,
            partition_level=1, max_node_num=0)
        end = time.time()
        nc = [pair[0]==pair[1] for pair in pairs_idx]
        times['ScalaGW'].append(end-start)
        results['ScalaGW'].append(np.mean(nc))
        error['ScalaGW'].append(calculate_infeat(coup_adj, p_s, p_t))

        #######BPG##############################################################################################################################
        ot_hyperpara_adj = {'loss_type': 'L2',
                            'ot_method': 'proximal',
                            'beta': 0.2,  #
                            'outer_iteration': 200,
                            'iter_bound': 1e-10,
                            'inner_iteration': 500,
                            'sk_bound': 1e-5,
                            'node_prior': 0,
                            'max_iter': 200,
                            'cost_bound': 1e-16,
                            'update_p': False,
                            'lr': 0,
                            'alpha': 0}
        start = time.time()
        coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(G_adj, G_adj_noise, p, q, ot_hyperpara_adj)
        end = time.time()
        times['BPG'].append(end-start)
        results['BPG'].append(node_correctness(coup_adj, np.eye(m)))
        error['BPG'].append(calculate_infeat(coup_adj, p, q))

        ######eBPG##############################################################################################################################
        # p = np.ones([m,1]).astype(np.float32)/m
        # q = np.ones([n,1]).astype(np.float32)/n
        # eps = 1e-2
        # if database == 'reddit':
        #     eps = 1e-1
        # start = time.time()
        # coup_adj = ot.gromov.entropic_gromov_wasserstein(G_adj, G_adj_noise, p.squeeze(-1), q.squeeze(-1),
        #                                                     loss_fun='square_loss', epsilon=eps,
        #                                                     verbose=True, log=False, max_iter=100)
        # end = time.time()
        # times['eBPG'].append(end-start)
        # results['eBPG'].append(node_correctness(coup_adj, np.eye(m)))
        # error['eBPG'].append(calculate_infeat(coup_adj, p, q))
        #########SpecGW#####################################################################################################################
        p = np.ones([m,1]).astype(np.float32)/m
        q = np.ones([n,1]).astype(np.float32)/n
        t = 10

        start = time.time()
        G_hk = sgw.undirected_normalized_heat_kernel(G,t)
        G_hk_noise = sgw.undirected_normalized_heat_kernel(G_noise,t)
        coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_noise, p.squeeze(), q.squeeze(),
                                                       loss_fun='square_loss', log=True)
        end = time.time()

        times['SpecGWL'].append(end-start)
        results['SpecGWL'].append(node_correctness(coup_hk, np.eye(m)))
        error['SpecGWL'].append(calculate_infeat(coup_hk, p, q))
        #
        for method, result in results.items():
            if len(results[method]):
                print('method: {} NC: {:.2f} Error: {:.2e}'.format(method, results[method][-1], error[method][-1]))

    print('---------------------------------Completed---------------------------------------')
    for method, result in results.items():
        print('Method: {} Acc: {:.4f}, Time: {:.4f}, Error: {:.4e}'.format(method,
            np.mean(results[method]), np.sum(times[method]), np.mean(error[method])))
        with open('result.txt', 'a+') as f:
            f.write('Data: {}, Noise:{}, Method: {}, Acc: {:.4f}, Time: {:.4f}, Error: {:.4e}\n'.format(
                database, noise_level, method,
                np.mean(results[method]), np.sum(times[method]), np.mean(error[method])))
