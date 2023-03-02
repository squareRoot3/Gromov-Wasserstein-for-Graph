import numpy as np
import networkx as nx
import random
import copy
import torch


def BAPG_torch(A, B, a=None, b=None, X=None, epoch=2000, eps=1e-5, rho=1e-1, min_rho=1e-1, scaling=1., early_stop=2000):
    if a is None:
        a = torch.ones([A.shape[0], 1]).float().cuda()/A.shape[0]
    if b is None:
        b = torch.ones([B.shape[0], 1]).float().cuda()/B.shape[0]
    if X is None:
        X = a@b.T
    obj_list, acc_list, res_list = [], [], []
    for ii in range(epoch):
        rho = max(rho/scaling, min_rho)
        X = X + 1e-10
        X = torch.exp(A@X@B/rho)*X
        X = X * (a / (X @ torch.ones_like(b)))
        X = torch.exp(A@X@B/rho)*X
        X = X * (b.T / (X.T @ torch.ones_like(a)).T)
        if ii > early_stop and ii % 50 == 0:
            objective = -torch.trace(A @ X @ B @ X.T)
            # print(ii, objective)
            if early_stop and len(obj_list) > 0 and (objective-obj_list[-1])/obj_list[-1] < eps:
                print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
    return X, obj_list


def BAPG_numpy(A, B, a=None, b=None, X=None, epoch=2000, eps=1e-5, rho=1e-1):
    if a is None:
        a = np.ones([A.shape[0], 1], dtype=np.float32)/A.shape[0]
    if b is None:
        b = np.ones([B.shape[0], 1], dtype=np.float32)/B.shape[0]
    if X is None:
        X = a@b.T
    obj_list, acc_list, res_list = [], [], []
    for ii in range(epoch):
        X = X + 1e-10
        X = np.exp(A@X@B/rho)*X
        X = X * (a / (X @  np.ones_like(b)))
        X = np.exp(A@X@B/rho)*X
        X = X * (b.T / (X.T @ np.ones_like(a)).T)
        if ii > 0 and ii % 50 == 0:
            objective = -np.trace(A @ X @ B @ X.T)
            # print(ii, objective)
            if len(obj_list) > 0 and np.abs((objective-obj_list[-1])/obj_list[-1]) < eps:
                print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
    return X, obj_list


def BPG_torch(cost_s, cost_t, p_s=None, p_t=None, trans0=None, beta=1e-1, error_bound=1e-10,
             outer_iter=200, inner_iter=100):
    a = torch.ones_like(p_s)/p_s.shape[0]
    if trans0 is None:
        trans0 = p_s @ p_t.T
    for oi in range(outer_iter):
        cost = - 2 * (cost_s @ trans0 @ cost_t.T)
        kernel = torch.exp(-cost / beta) * trans0
        for ii in range(inner_iter):
            b = p_t / (kernel.T@a)
            a_new = p_s / (kernel@b)
            relative_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
            a = a_new
            if relative_error < 1e-10:
                break
        trans = (a @ b.T) * kernel
        relative_error = torch.sum(torch.abs(trans - trans0)) / torch.sum(torch.abs(trans0))
        if relative_error < error_bound:
            break
        trans0 = trans
        if oi % 50 == 0 and oi > 0:
            print(oi, -torch.trace(cost_s @ trans @ cost_t @ trans.T))
    return trans


def add_noisy_edges(graph: nx.graph, noisy_level: float) -> nx.graph:
    nodes = list(graph.nodes)
    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges)
    graph_noisy = copy.deepcopy(graph)
    if num_noisy_edges > 0:
        i = 0
        while i < num_noisy_edges:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if (src, dst) not in graph_noisy.edges:
                graph_noisy.add_edge(src, dst)
                i += 1
    return graph_noisy


def add_noisy_nodes(graph: nx.graph, noisy_level: float) -> nx.graph:
    num_nodes = len(graph.nodes)
    num_noisy_nodes = int(noisy_level * num_nodes)

    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges / num_nodes + 1)

    graph_noisy = copy.deepcopy(graph)
    if num_noisy_nodes > 0:
        for i in range(num_noisy_nodes):
            graph_noisy.add_node(int(i + num_nodes))
            j = 0
            while j < num_noisy_edges:
                src = random.choice(list(range(i + num_nodes)))
                if (src, int(i + num_nodes)) not in graph_noisy.edges:
                    graph_noisy.add_edge(src, int(i + num_nodes))
                    j += 1
    return graph_noisy


def node_correctness(coup, perm_inv):
    coup_max = coup.argmax(1)
    perm_inv_max = perm_inv.argmax(1)
    acc = np.sum(coup_max == perm_inv_max) / len(coup_max)
    return acc


def calculate_infeat(X, a, b):
    gap = np.linalg.norm((X.sum(0) - b.squeeze(-1))) + np.linalg.norm(X.sum(1) - a.squeeze(-1))
    return gap


def get_partition(coup):
    est_idx = np.argmax(coup, axis=1)
    num_clusters = np.max(est_idx)
    partition = []
    for j in range(num_clusters + 1):
        partition.append(set(np.argwhere(est_idx == j).T[0]))
    return partition