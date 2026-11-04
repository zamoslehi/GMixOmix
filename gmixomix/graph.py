"""Graph construction."""
from sklearn.neighbors import NearestNeighbors


def compute_adjaciency_matrix(X, n_neighbors):
    """Compute a nearest neighbours adjacency matrix."""
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    return nn.kneighbors_graph(X).toarray()
