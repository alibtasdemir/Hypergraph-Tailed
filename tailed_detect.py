class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_neighbors(self, node):
        neighs = []
        for edge in self.edges:
            if node in edge:
                edge = list(edge)
                edge.remove(node)
                neighs.append(edge[0])
        return set(neighs)

    def get_tailed_triangles(self, clique):
        clique = set(clique)
        exp_n = set(clique.copy())
        removed = set()
        c_nodes = set(clique.copy())
        for node in c_nodes:
            n_neig = self.get_neighbors(node)
            rem = n_neig.intersection(exp_n)
            removed.update(rem)
            exp_n -= removed
            n_neig -= removed
            exp_n.update(n_neig)
        clique, exp_n = list(clique), list(exp_n)
        del (removed, c_nodes, rem, n_neig)
        return [tuple(clique + [x]) for x in list(exp_n)]


G = [1, 2, 3, 4, 5, 6, 7]
edges = [(1, 2), (1, 3), (1, 7), (2, 7), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (4, 6)]

g = Graph(G, edges)
del (G, edges)

clique = set([2, 4, 6])
print(g.get_tailed_triangles(clique))
