import os
from itertools import combinations
import networkx as nx

from config import *


def get_hedges_and_weights(config, graph_name):
    timestamped_hedges, unique_hedges, weights = [], [], []
    gpath = os.path.join(config.input_folder, graph_name, "{}-hypergraph.hg".format(graph_name))

    print("Generating timestamped hedges")
    with open(gpath, 'r') as graph:
        for entry in graph.readlines()[1:]:
            id, nodes, time = entry.split(';')
            hedge = [int(i) for i in nodes.split(',')]
            hedge.sort()
            timestamped_hedges.append(tuple(hedge))

    hedges = sorted(timestamped_hedges)
    print("Generating unique hedges...")
    w, last_hedge = None, None
    for idx, hedge in enumerate(hedges):
        if idx == 0:
            last_hedge = hedge
            w = 1
            continue
        if hedge == last_hedge:
            w += 1
            continue
        unique_hedges.append(last_hedge)
        weights.append(w)
        last_hedge = hedge
        w = 1

    unique_hedges.append(last_hedge)
    weights.append(w)
    print("..done")

    return timestamped_hedges, unique_hedges, weights


def hedges_to_pg(hedges, weights, p):
    print("Generating {}-pg..".format(p))
    assert (p >= 2)
    pg = nx.Graph()
    node_size = p - 1
    n_edges = 0
    for idx, hedge in enumerate(hedges):
        if len(hedge) < p:
            continue
        w = weights[idx]
        if p == 2:
            nodes = hedge
        else:
            nodes = []
            for node in combinations(hedge, node_size):
                nodes.append(node)

        # Update edges
        for edge in combinations(nodes, 2):
            node1, node2 = edge
            if p > 2:
                if len(set(node1 + node2)) > p:
                    continue
            if pg.has_edge(node1, node2):
                pg[node1][node2]['weight'] += w
            else:
                pg.add_edge(node1, node2, weight=w)
                n_edges += 1
                if n_edges % (500 * 1000) == 0:
                    print(n_edges)
                if n_edges == 500 * 1000 * 1000:  # cannot handle more
                    print("Cannot handle more")
                    return pg

    print("..done")
    return pg


def hedges_to_supcount(hedges, weights, set_size):
    print("Getting supcounts..")
    supcount_dict = {}
    for idx, hedge in enumerate(hedges):
        w = weights[idx]
        for itemset in combinations(hedge, set_size):
            if itemset in supcount_dict:
                supcount_dict[itemset] += w
            else:
                supcount_dict[itemset] = w

    print("..done")
    return supcount_dict


def hedge_to_pg_nodes(hedge, p):
    nodes = []
    if p == 2:
        nodes = list(hedge)
    else:
        for combination in combinations(hedge, p - 1):
            nodes.append(combination)
    return nodes
