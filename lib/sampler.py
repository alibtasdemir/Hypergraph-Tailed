import random
from config import *


def get_cliques(size, G, exclude, n, strong=False, max_iter=10*1000*10):
    """
    :param size: clique size
    :param G: nx graph
    :param exclude: exclude these
    :param n: try to find n cliques
    :param strong: strong clique
    :param max_iter: try until this iteration of while loop
    :return:
    """

    cliques = set()

    n_neighbors = size - 1
    n_edges = size * (size - 1) / 2

    nodes = list(G.nodes)

    n_iter = 0
    while len(cliques) < n:
        if len(cliques) >= n:
            break
        if n_iter >= max_iter:
            break
        if n_iter % (10 * 1000) == 0:
            print("n_iter: {}".format(n_iter))

        n_iter += 1

        node = random.choice(nodes)  # select a random node

        all_neighbors = [neigh for neigh in G.neighbors(node)]
        if len(all_neighbors) < n_neighbors:
            continue

        # find at most 1 clique
        for i in range(10):

            neighbors = random.sample(all_neighbors, k=n_neighbors)

            subn = [node] + neighbors
            subg = G.subgraph(subn)

            if subg.number_of_edges() != n_edges:
                continue  # not a clique

            clique = subn

            clique.sort()
            clique = tuple(clique)

            if clique not in exclude:
                cliques.add(clique)
                break

    cliques = list(cliques)  # list of non-overlapping cliques

    print("# node iterations {}".format(n_iter))
    print("# cliques found: {}".format(len(cliques)))

    return cliques


def get_hubs(size, G, exclude, n, induced=False, max_iter=10*1000*10):
    hubs = set()
    n_neighbors = size - 1
    nodes = list(G.nodes)
    n_iter = 0
    while len(hubs) < n:
        if len(hubs) >= n:
            break
        if n_iter > max_iter:
            break
        n_iter += 1

        node = random.choice(nodes)
        all_neighbors = [neighbor for neighbor in G.neighbors(node)]
        if len(all_neighbors) < n_neighbors:
            continue
        for i in range(min(20, G.degree[node] // 4)):
            neighbors = random.sample(all_neighbors, k=n_neighbors)
            hub = [node] + neighbors
            # Not induced
            if induced:
                subg = G.subgraph(hub)
                if subg.number_of_edges != n_neighbors:
                    continue    # Not induced
            hub.sort()
            hub = tuple(hub)
            if hub not in exclude:
                hubs.add(hub)
                if len(hubs) >= n:
                    break
    hubs = list(hubs)

    print("# node iterations {}".format(n_iter))
    print("# hubs found: {}".format(len(hubs)))

    return hubs


def get_tailed(size, G, exclude, n, max_iter=10*1000*10):
    tailed_figures = set()

    n_neighbors = size - 1
    n_edges = size * (size - 1) / 2

    nodes = list(G.nodes)

    n_iter = 0
    while len(tailed_figures) < n:
        if len(tailed_figures) >= n:
            break
        if n_iter >= max_iter:
            break
        if n_iter % (10 * 1000) == 0:
            print("n_iter: {}".format(n_iter))

        n_iter += 1

        node = random.choice(nodes)  # select a random node

        all_neighbors = [neigh for neigh in G.neighbors(node)]
        if len(all_neighbors) < n_neighbors:
            continue

        # find at most 1 clique
        for i in range(10):

            neighbors = random.sample(all_neighbors, k=n_neighbors)

            subn = [node] + neighbors
            subg = G.subgraph(subn)

            if subg.number_of_edges() != n_edges:
                continue  # not a clique

            # Search for a tail
            current_clique = set(subn)
            expansion_list = set(current_clique.copy())
            remove_list = set()
            current_nodes =set(current_clique.copy())

            for node in current_nodes:
                cn_neighbors = set([neigh for neigh in G.neighbors(node)])
                rem = cn_neighbors.intersection(expansion_list)
                remove_list.update(rem)
                expansion_list -= remove_list
                cn_neighbors -= remove_list
                expansion_list.update(cn_neighbors)
            if len(expansion_list) == 0:
                continue
            clique, expansion_list = list(current_clique), list(expansion_list)

            for tailed in [tuple(clique + [x]) for x in list(expansion_list)]:
                tailed = tuple(sorted(list(tailed)))
                if tailed not in exclude:
                    tailed_figures.add(tailed)
                    if len(tailed_figures) >= n:
                        break

    tailed_figures = list(tailed_figures)  # list of non-overlapping cliques

    print("# node iterations {}".format(n_iter))
    print("# {}-tailed found: {}".format(size, len(tailed_figures)))

    return tailed_figures


if __name__ == "__main__":
    import networkx as nx
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 7)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 6)
    G.add_edge(2, 7)
    G.add_edge(3, 4)
    G.add_edge(4, 6)
    TF = get_tailed(3, G, [], 1)
    print(TF)
