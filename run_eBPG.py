import time
import pickle
import warnings
import ot
import os
import argparse
from collections import defaultdict
from BAPG import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='proteins', help='proteins / reddit / enzymes / synthetic')
parser.add_argument('--noise_level', type=float, default=0.)
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--loss_fun', type=str, default='square_loss', help='square_loss/kl_loss')
parser.add_argument('--eps', type=float, default=[0.1, 0.01, 0.001], nargs='+')

args = parser.parse_args()
eps_list = args.eps
database = args.dataset
noise_level = args.noise_level
use_gpu = args.use_gpu
print(eps_list)

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
    with open('data/Random/Graph1.pk', 'rb') as f:
        graph_pairs = pickle.load(f)
        for num_node in [500, 1000, 1500, 2000, 2500]:
            for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                for G, G_noise in graph_pairs[(num_node, noise_level)]:
                    graphs.append(G)
                    noise_graphs.append(G_noise)

noise_graphs = []
if noise_level > 1e-6:
    for G_src in graphs:
        G_dst = add_noisy_edges(G_src, noise_level)
        G_dst = add_noisy_nodes(G_dst, noise_level)
        noise_graphs.append(G_dst)
else:
    noise_graphs = graphs

total_num_graphs = len(graphs)
print('total_num_graphs: ', total_num_graphs)

results, times, error = defaultdict(list), defaultdict(list), defaultdict(list)
for eps in eps_list:
    for j in range(0, total_num_graphs):  ## total_num_graphs
        print('graph id: ', j)
        G = graphs[j]
        G_noise = noise_graphs[j]
        G_adj = nx.to_numpy_array(G).astype(np.float64)
        G_adj_noise = nx.to_numpy_array(G_noise).astype(np.float64)
        m, n = G_adj.shape[0], G_adj_noise.shape[0]
        p = np.ones([m, 1], dtype=G_adj.dtype) / m
        q = np.ones([n, 1], dtype=G_adj.dtype) / n
        if use_gpu:
            a = torch.ones([m, 1]).cuda()/m
            b = torch.ones([n, 1]).cuda()/n
            G_adj_gpu = torch.tensor(G_adj.astype(np.float32)).cuda()
            G_adj_noise_gpu = torch.tensor(G_adj_noise.astype(np.float32)).cuda()
            start = time.time()
            coup_adj, _ = ot.gromov.entropic_gromov_wasserstein(G_adj_gpu, G_adj_noise_gpu, a.squeeze(-1), b.squeeze(-1),
                                                                loss_fun=args.loss_fun, epsilon=eps,
                                                                verbose=False, log=True, max_iter=100, tol=1e-6)
            end = time.time()
            coup_adj = coup_adj.cpu().double().numpy()
        else:

            start = time.time()
            coup_adj, _ = ot.gromov.entropic_gromov_wasserstein(G_adj, G_adj_noise, p.squeeze(-1), q.squeeze(-1),
                                                                loss_fun=args.loss_fun, epsilon=eps,
                                                                verbose=False, log=True, max_iter=100, tol=1e-6)
            end = time.time()
        times[eps].append(end-start)
        results[eps].append(node_correctness(coup_adj, np.eye(m)))
        error[eps].append(calculate_infeat(coup_adj, p, q))
        print('eps: {} Mean: {:.4f}, Time: {:.4f}, Error: {:.4e}'.format(
            eps, results[eps][-1], times[eps][-1], error[eps][-1]))

print('---------------------------------Completed---------------------------------------')
for eps, result in results.items():
    print('eps: {} Mean: {:.4f}, Time: {:.4f}, Error: {:.4e}'.format(
        eps, np.mean(results[eps]), np.sum(times[eps]), np.mean(error[eps])))
    with open('result.txt', 'a+') as f:
        f.write('Method: eBPG, Data:{}, eps:{}, Acc: {:.4f}, Time: {:.4f}, Error: {:.4e}\n'.format(database, eps,
            np.mean(results[eps]), np.sum(times[eps]), np.mean(error[eps])))
