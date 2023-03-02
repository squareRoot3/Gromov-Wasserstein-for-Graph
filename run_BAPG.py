from collections import defaultdict
import pickle
from BAPG import *
import torch
import time
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def defaul():
    return defaultdict(list)


def BAPG_torch(A, B, a=None, b=None, X=None, epoch=200, eps=1e-5, rho=1e-1):
    if a is None:
        a = torch.ones([A.shape[0], 1], dtype=A.dtype).cuda()/A.shape[0]
    if b is None:
        b = torch.ones([B.shape[0], 1], dtype=A.dtype).cuda()/B.shape[0]
    if X is None:
        X = a@b.T
    obj_list, obj1_list, obj2_list, gap_list = [], [], [], []
    for ii in range(epoch):
        X = X + 1e-10
        X = torch.exp(A@X@B/rho)*X
        pi = X * (a / (X @ torch.ones_like(b)))
        X = torch.exp(A@pi@B/rho)*pi
        X = X * (b.T / (X.T @ torch.ones_like(a)).T)
        if (ii+1) % 50 == 0:
            objective = -torch.trace(A @ X @ B @ X.T).item()
            X2 = (X+pi)/2
            gap = (X2.sum(0)-b.squeeze(-1)).norm() + (X2.sum(1)-a.squeeze(-1)).norm()
            gap_list.append(gap.item())
            if len(obj_list) > 0 and np.abs((objective-obj_list[-1])/obj_list[-1]) < eps:
                print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
    return X2, obj_list, gap_list


def BAPG_numpy(A, B, a=None, b=None, X=None, epoch=200, eps=1e-5, rho=1e-1):
    if a is None:
        a = np.ones([A.shape[0], 1], dtype=A.dtype)/A.shape[0]
    if b is None:
        b = np.ones([B.shape[0], 1], dtype=A.dtype)/B.shape[0]
    if X is None:
        X = a@b.T
    obj_list, gap_list = [], []
    for ii in range(epoch):
        X = X+1e-10
        X = np.exp(A@X@B/rho)*X
        pi = X * (a / (X @ np.ones_like(b)))
        X = np.exp(A@pi@B/rho)*pi
        X = X * (b.T / (X.T @ np.ones_like(a)).T)
        if (ii+1) % 50 == 0:
            X2 = (X+pi)/2
            gap = np.linalg.norm((X2.sum(0)-b.squeeze(-1))) + np.linalg.norm(X2.sum(1)-a.squeeze(-1))
            gap_list.append(gap.item())
            objective = -np.trace(A @ X @ B @ X.T)
            if len(obj_list) > 0 and np.abs((objective-obj_list[-1])/obj_list[-1]) < eps:
                print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
    return X2, obj_list, gap_list
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='proteins', help='proteins / reddit / enzymes / synthetic')
parser.add_argument('--noise_level', type=float, default=0.)
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--loss_fun', type=str, default='square_loss', help='square_loss/kl_loss')
parser.add_argument('--rho', type=float, default=[0.5, 0.2, 0.1, 0.05, 0.01], nargs='+')

args = parser.parse_args()
rho_list = args.rho
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
        print(graph_pairs)
        for num_node in [500, 1000, 1500, 2000, 2500]:
            for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                for G, G_noise in graph_pairs[(num_node, noise_level)]:
                    graphs.append(G)
                    noise_graphs.append(G_noise)

if database != 'synthetic':
    if noise_level > 0:
        noise_graphs = []
        for G_src in graphs:
            G_dst = add_noisy_edges(G_src, noise_level)
            G_dst = add_noisy_nodes(G_dst, noise_level)
            noise_graphs.append(G_dst)
    else:
        noise_graphs = graphs


gap_rho = defaultdict(list)
acc_rho = defaultdict(list)
time_rho = defaultdict(list)
gap2_rho = defaultdict(list)
acc2_rho = defaultdict(list)
time2_rho = defaultdict(list)
for j in range(len(graphs)):
    print(j)
    G = graphs[j]
    G_noise = noise_graphs[j]
    G_adj = nx.to_numpy_array(G).astype(np.float32)
    G_adj_noise = nx.to_numpy_array(G_noise).astype(np.float32)
    G_adj_gpu = torch.tensor(G_adj).cuda()
    G_adj_noise_gpu = torch.tensor(G_adj_noise).cuda()
    m, n = G_adj.shape[0], G_adj_noise.shape[0]
    for rho in rho_list:  #0.5,0.2,0.1,0.05,
        epoch = 2000 if rho < 0.2 else 4000
        start = time.time()
        # coup_bap, obj_list, gap_list = BAPG_numpy(
        #     A=G_adj, B=G_adj_noise, epoch=2000, rho=rho, eps=1e-5, scaling=1.01, max_rho=0.5)
        if args.use_gpu:
            coup_bap, obj_list, gap_list = BAPG_torch(
                A=G_adj_gpu, B=G_adj_noise_gpu, epoch=epoch, rho=rho, eps=1e-5)
            end = time.time()
            acc = node_correctness(coup_bap.cpu().numpy(), np.eye(m))
            print(rho, gap_list[-1], acc)
            acc_rho[rho].append(acc)
            time_rho[rho].append(end-start)
            gap_rho[rho].append(gap_list[-1])
            coup_bap, obj_list, gap_list = BAPG_torch(
                A=G_adj_gpu, B=G_adj_noise_gpu, X=coup_bap, epoch=100, rho=100, eps=1e-5)
            end = time.time()
            acc = node_correctness(coup_bap.cpu().numpy(), np.eye(m))
            print('100', gap_list[-1], acc)
        else:
            coup_bap, obj_list, gap_list = BAPG_numpy(
                A=G_adj, B=G_adj_noise, epoch=2000, rho=rho, eps=1e-5)
            end = time.time()
            acc = node_correctness(coup_bap, np.eye(m))
            print(rho, gap_list[-1], acc)
            acc_rho[rho].append(acc)
            time_rho[rho].append(end-start)
            gap_rho[rho].append(gap_list[-1])
            coup_bap, obj_list, gap_list = BAPG_numpy(
                A=G_adj, B=G_adj_noise, X=coup_bap, epoch=100, rho=100, eps=1e-5)
            end = time.time()
            acc = node_correctness(coup_bap, np.eye(m))
            print('100', gap_list[-1], acc)

        acc2_rho[rho].append(acc)
        time2_rho[rho].append(end-start)
        gap2_rho[rho].append(gap_list[-1])


for rho in rho_list:
    print('rho:{},gap:{:.2e},acc:{:.2f},time:{:.1f},gap2:{:.2e},acc2:{:.2f},time2:{:.1f}'.format(
        rho, np.mean(gap_rho[rho]), np.mean(acc_rho[rho])*100, np.sum(time_rho[rho]),
        np.mean(gap2_rho[rho]), np.mean(acc2_rho[rho])*100, np.sum(time2_rho[rho])))

    with open('result.txt', 'a+') as f:
        f.write('Data:{},Noise:{},rho:{},gap:{:.2e},acc:{:.2f},time:{:.1f},gap2:{:.2e},acc2:{:.2f},time2:{:.1f}\n'.format(
            database, noise_level, rho, np.mean(gap_rho[rho]), np.mean(acc_rho[rho])*100, np.sum(time_rho[rho]),
            np.mean(gap2_rho[rho]), np.mean(acc2_rho[rho])*100, np.sum(time2_rho[rho])))
