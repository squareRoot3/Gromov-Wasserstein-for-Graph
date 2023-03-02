import time
import ot
import utils.spectralGW as sgw
from utils.GromovWassersteinFramework import *
import utils.GromovWassersteinGraphToolkit as GwGt
from BAPG import *
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings("ignore")
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

num_nodes = [500, 1000, 1500, 2000, 2500]
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

with open('data/Random/graph1.pk', 'rb') as f:
    graph_pairs = pickle.load(f)

print('Total generated random graph number: ', len(graph_pairs))

def defaul():
    return defaultdict(list)
results = defaultdict(defaul)
times = defaultdict(defaul)

for num_node in num_nodes:
    for noise_level in noise_levels:
        for G, G_noise in graph_pairs[(num_node, noise_level)]:  ## total_num_graphs
            print(num_node, noise_level)
            G_adj = nx.to_numpy_array(G).astype(np.float32)
            G_adj_noise = nx.to_numpy_array(G_noise).astype(np.float32)
            m, n = G_adj.shape[0], G_adj_noise.shape[0]
            p = np.ones([m,1]).astype(np.float32)/m
            q = np.ones([n,1]).astype(np.float32)/n
            Xinit = p@q.T
            #########Our-BAPG-GPU##############################################################################################################################
            G_adj_gpu = torch.tensor(G_adj).cuda()
            G_adj_noise_gpu = torch.tensor(G_adj_noise).cuda()
            start = time.time()
            rho = 0.1
            coup_bap, obj_list_bap = BAPG_torch(A=G_adj_gpu, B=G_adj_noise_gpu, a=None, b=None, epoch=1000, eps=1e-5,
                                                rho=rho, min_rho=rho)
            end = time.time()
            times['BAPG'][(num_node, noise_level)].append(end-start)
            results['BAPG'][(num_node, noise_level)].append(node_correctness(coup_bap.cpu().numpy(), np.eye(m)))
            #######Our-BAPG-CPU##############################################################################################################################
            # start = time.time()
            # rho = 0.1
            # coup_bap, obj_list_bap = BAPG_numpy(A=G_adj, B=G_adj_noise, a=p, b=q, X=Xinit, epoch=500, eps=1e-5,
            #                                     rho=rho)
            # end = time.time()
            # times['BAPGcpu'][(num_node, noise_level)].append(end-start)
            # results['BAPGcpu'][(num_node, noise_level)].append(node_correctness(coup_bap, np.eye(m)))
            ######FW###########################################################################################################################
            p = np.ones([m,1]).astype(np.float32)/m
            q = np.ones([n,1]).astype(np.float32)/n
            start = time.time()
            coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_adj, G_adj_noise, p.squeeze(), q.squeeze(),
                                                           loss_fun='kl_loss', log=True)
            end = time.time()
            times['FW'][(num_node, noise_level)].append(end-start)
            results['FW'][(num_node, noise_level)].append(node_correctness(coup_hk, np.eye(m)))
            #########BPG-S##############################################################################################################################
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
            coup_adj, d_gw, p_s = gromov_wasserstein_discrepancy(G_adj, G_adj_noise, p_s, p_t, ot_hyperpara_adj)
            end = time.time()
            times['BPG-S'][(num_node, noise_level)].append(end-start)
            results['BPG-S'][(num_node, noise_level)].append(node_correctness(coup_adj, np.eye(m)))

            ##ScalaGW##############################################################################################################################
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
                weights=None, predefine_barycenter=False, max_node_num=0)
            end = time.time()
            nc = [pair[0]==pair[1] for pair in pairs_idx]
            times['ScalaGW'][(num_node, noise_level)].append(end-start)
            results['ScalaGW'][(num_node, noise_level)].append(np.mean(nc))

            ##BPG##############################################################################################################################
            ot_hyperpara_adj = {'loss_type': 'L2',
                                'ot_method': 'proximal',
                                'beta': 0.2,  # 2
                                'outer_iteration': 500,
                                'iter_bound': 1e-10,
                                'inner_iteration': 100,
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
            times['BPG'][(num_node, noise_level)].append(end-start)
            results['BPG'][(num_node, noise_level)].append(node_correctness(coup_adj, np.eye(m)))

            ####eBPG##############################################################################################################################
            p = np.ones([m,1]).astype(np.float32)/m
            q = np.ones([n,1]).astype(np.float32)/n
            start = time.time()
            coup_adj, _ = ot.gromov.entropic_gromov_wasserstein(G_adj, G_adj_noise, p.squeeze(-1), q.squeeze(-1),
                                                                loss_fun='square_loss', epsilon=1e-2,
                                                                verbose=False, log=True, max_iter=200)
            end = time.time()
            times['eBPG'][(num_node, noise_level)].append(end-start)
            results['eBPG'][(num_node, noise_level)].append(node_correctness(coup_adj, np.eye(m)))

            ##SpecGW#####################################################################################################################
            start = time.time()
            t = 10
            G_hk = sgw.undirected_normalized_heat_kernel(G,t)
            G_hk_noise = sgw.undirected_normalized_heat_kernel(G_noise,t)
            coup_hk, log_hk = ot.gromov.gromov_wasserstein(G_hk, G_hk_noise, p.squeeze(), q.squeeze(),
                                                           loss_fun='square_loss', log=True)
            end = time.time()

            times['SpecGW'][(num_node, noise_level)].append(end-start)
            results['SpecGW'][(num_node, noise_level)].append(node_correctness(coup_hk, np.eye(m)))

            for method, result in results.items():
                if len(results[method]):
                    print('method: {} NC: {:.4f}, Time: {:.4f}'.format(method,
                                                                       results[method][(num_node, noise_level)][-1],
                                                                       times[method][(num_node, noise_level)][-1]))
print(results, times)
