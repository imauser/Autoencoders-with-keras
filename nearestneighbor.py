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


def find_nearest_class(neighbor, neighborhood, neighborhood_labels):
    """ calculates lowest distance to a dictionary
    >>> find_nearest_class([1,1],[[1,2],[5,5],[2,2]], [1,2,3])
    1
    >>> find_nearest_class([1,1],[[1,2], [1,2], [1,2],[5,5],[2,2]], [1, 1, 1 ,2,3])
    1
    """
    classes = dict()
    nof_classes = dict()
    for i in range(len(neighborhood)):
        act_class = 0
        if hasattr(neighborhood_labels[i], '__contains__'):
            act_class = neighborhood_labels[i][0]
        else:
            act_class = neighborhood_labels[i]

        if act_class in classes:
            classes[act_class] += neighbor_distance(neighbor, neighborhood[i])
            nof_classes[act_class] += 1
        else:
            classes[act_class] = neighbor_distance(neighbor, neighborhood[i])
            nof_classes[act_class] = 1
    from sys import maxsize
    key, smallest = -1, maxsize

    for k,v in classes.iteritems():
        mean = v/nof_classes[k]
        if mean < smallest:
            smallest = mean
            key = k

    return key

