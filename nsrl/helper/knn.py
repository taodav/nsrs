import numpy as np

def euclidian_dist(batch_x1, x2):
    """
    Euclidian distance
    Parameters
    ----------
    batch_x1: numpy array of [batch_size, dim]
    x2: numpy array of [dim]

    Returns
    -------
    euclidian distance [batch_size]
    """
    return np.sum(np.square(batch_x1 - x2), axis=-1)

def batch_count_scaled_knn(batch_states, memory, k=10, d=euclidian_dist):
    """
    Computes KNN scores for top k closest unique elements, scaling each k by
    1 / (sqrt(n_i)) for i \in k.
    ----------
    k: k neighbors
    batch_states: numpy array of size [batch_size x state_size]
    memory: numpy array of size [memory_size x state_size]
    d: distance function with signature R^d, R^d -> R

    Returns
    -------
    array of size batch_size x k
    """
    unsorted_dists = np.stack([d(batch_states, m) for m in memory], axis=-1)
    dists = np.sort(unsorted_dists, axis=-1)
    scores = []
    for d in dists:
        d_elements, counts = np.unique(d, return_counts=True)
        scales = 1 / np.sqrt(counts)
        scaled_dists = np.multiply(d_elements, scales)
        scores.append(scaled_dists[:k])
    return np.array(scores)

def batch_knn(batch_states, memory, k=10, d=euclidian_dist):
    """
    Computes KNN scores for top k closest elements.
    Parameters
    ----------
    k: k neighbors
    batch_states: numpy array of size [batch_size x state_size]
    memory: numpy array of size [memory_size x state_size]
    d: distance function with signature R^d, R^d -> R

    Returns
    -------
    array of size batch_size x k
    """
    unsorted_dists = np.stack([d(batch_states, m) for m in memory], axis=-1)
    dists = np.sort(unsorted_dists, axis=-1)
    return dists[:, :k]

def avg_knn_scores(batch_states, memory, k=10, knn=batch_count_scaled_knn):
    """
    Computes average KNN score for each element in batch of states
    Parameters
    ----------
    k: k neighbors
    batch_states: numpy array of size [batch_size x state_size]
    memory: numpy array of size [memory_size x state_size]

    Returns
    -------
    numpy array of scores of dims [batch_size]
    """
    nearest_neighbor_scores = knn(batch_states, memory, k=k)
    return np.average(nearest_neighbor_scores[:, 1:], axis=-1)

def ranked_avg_knn_scores(batch_states, memory, k=10, knn=batch_count_scaled_knn):
    """
    Computes ranked average KNN score for each element in batch of states
    \sum_{i = 1}^{K} (1/i) * d(x, x_i)
    Parameters
    ----------
    k: k neighbors
    batch_states: numpy array of size [batch_size x state_size]
    memory: numpy array of size [memory_size x state_size]

    Returns
    -------
    numpy array of scores of dims [batch_size]
    """
    nearest_neighbor_scores = knn(batch_states, memory, k=k)
    k = nearest_neighbor_scores.shape[1]
    scales = 1 / np.expand_dims(np.arange(1, k + 1), axis=0).repeat(batch_states.shape[0], axis=0)
    # There may be the edge case where the number of unique distances for this particular batch
    # is less than k. If that's the case, we need to reduce our scales dimension.
    # This means one of two things:
    # 1. you either have a very small map, or
    # 2. your representation has collapsed into less than k points.
    ranked_avg_scores = np.multiply(nearest_neighbor_scores, scales)
    return np.sum(ranked_avg_scores, axis=-1)

def weighted_k_nearest(batch_states, memory, k=1):
    """
    Computes weight KNN scores for state s based on:
    nearest_neighbors_dist(s, memory, k) / count(s, memory)
    Parameters
    ----------
    k: k neighbors
    batch_states: numpy array of size [batch_size x state_size]
    memory: numpy array of size [memory_size x state_size]
    d: distance function with signature R^d, R^d -> R

    Returns
    -------

    """
    unsorted_dists = np.stack([euclidian_dist(batch_states, m) for m in memory], axis=-1)
    batch_dists = np.sort(unsorted_dists, axis=-1)
    non_zero_idx = np.array([next(i for i, d in enumerate(dists) if d != 0) for dists in batch_dists])
    counts = non_zero_idx
    indices = np.array([np.arange(i, i + k) for i in non_zero_idx])
    top_k_nonzero = np.take(batch_dists, indices)
    unnormalized_scores = np.sum(top_k_nonzero, axis=-1)
    scores = np.divide(unnormalized_scores, counts)

    return scores

# Unit tests
if __name__ == "__main__":

    batch = np.array([
        [1, 1], [2, 2], [3, 3], [1, 1]
    ])

    memory = np.array([
        [0, 0], [1, 1], [2, 2], [1, 1]
    ])

    res = batch_knn(batch, memory, k=3)

    target = np.array([
        [0, 2, 2], [0, 2, 8], [2, 8, 18]
    ])
    assert np.array_equal(res, target)
    print("done")