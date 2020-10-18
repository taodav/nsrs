import numpy as np
import matplotlib.pyplot as plt

def tsne(data, n_comps, plot=False):
    
    pass

def pca(data, n_comps, plot=False):
    """
    PCA dimensionality reduction.
    :param data: batch x original_dim
    :param n_comps: int, number of components to keep
    :return: batch x n_comps numpy array
    """
    orig_dim = data.shape[-1]
    mean = np.average(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev

    cov = np.cov(normalized_data.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov)

    if plot:
        # calculate cumulative sum of explained variances
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        # plot explained variances
        plt.bar(range(1, orig_dim + 1), var_exp, alpha=0.5,
                align='center', label='individual explained variance')
        plt.step(range(1, orig_dim + 1), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.show()

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    w = np.hstack(
        [eigen_pairs[i][1][:, np.newaxis] for i in range(n_comps)]
    )

    return normalized_data.dot(w)