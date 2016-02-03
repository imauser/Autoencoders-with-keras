import numpy as np

def neighbor_distance(x1, x2):
    """ :return: distance of the two scalars or matrices in l2/frobenius norm
    """
    return np.linalg.norm(np.subtract(x1, x2))


def find_nearest_neighbor_index_vectorized(neighbor, neighborhood):
    """ Calculates the nearest neighbor for a given array
    >>> find_nearest_neighbor_index_vectorized([1,1],[[1,2], [2,2], [0,0]])
    0
    """
    f = np.vectorize(neighbor_distance, excluded='x2')
    return np.argmin(f(neighborhood, neighbor).T)


def find_nearest_neighbor_index(neighbor, neighborhood):
    """ Calculates the nearest neighbor for a given array
    >>> find_nearest_neighbor_index([1,1],[[1,2], [2,2], [1,1]])
    2
    """
    nearest_neighbor = (0, np.inf)  # (index, distance)
    for i, n in enumerate(neighborhood):
        n_dist = neighbor_distance(neighbor, n)
        if n_dist < nearest_neighbor[1]:
            nearest_neighbor = (i, n_dist)
    return nearest_neighbor[0]
