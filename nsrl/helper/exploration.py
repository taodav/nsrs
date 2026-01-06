"""
Exploration helpers. Requires PyTorch
"""
import torch
from nsrl.helper.knn import ranked_avg_knn_scores, avg_knn_scores, batch_knn, batch_count_scaled_knn
from nsrl.helper.pytorch import device, calculate_large_batch

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from ..learning_algos.NSRS_pytorch import NSRS


def calculate_unpredictability_estimate(states, target_network, predictor_network):
    """
    Calculating unpredictability of given states.

    Parameters
    ----------
    states: States to calculate scores from of size [batch_size x (state_dim)]
    target_network: randomly initialized fixed encoder network
    predictor_network: network to try and predict predictor network.

    Returns
    -------
    Scores of size [batch_size]
    """
    pass


def calculate_scores(states, memory, encoder=None, k=10, dist_score=ranked_avg_knn_scores,
                     knn=batch_count_scaled_knn, plotter=None, _count = None):
    """
    Calculating KNN scores for each of the states. We want to
    optionally encode ALL the states in the buffer to calculate things.

    Parameters
    ----------
    states: States to calculate scores from of size [batch_size x (state_dim)]
    encoder: Encoder that takes in [batch_size x (state_dim)] and returns [batch_size x encoded_size]

    Returns
    -------
    Scores of size [batch_size]
    """
    # don't calculate gradients here!
    print("states shape: ", states.shape)
    print("memory shape: ", memory.shape)


    with torch.no_grad():
        if encoder is None:
            encoder = lambda x: x

        # one big bad batch
        # encoded_memory = encoder(torch.tensor(memory, dtype=torch.float).to(device))
        encoded_memory = calculate_large_batch(encoder, memory)

        encoded_states = states
        # REFACTOR THIS
        if encoded_states.shape[-1] != encoded_memory.shape[-1]:
            # encoded_states = encoder(torch.tensor(states, dtype=torch.float).to(device))
            encoded_states = calculate_large_batch(encoder, states)
        scores = dist_score(encoded_states.cpu().detach().numpy(),
                            encoded_memory.cpu().detach().numpy(), k=k,
                            knn=knn)
        if encoded_states.shape[0] != 1 and plotter is not None:
            print("encoded_states shape: ", encoded_states.shape)
            plotter.plot("newest knn scores", np.array([_count]), np.array([scores[-1]]), "newest knn scores")
            plotter.plot("knn scores 0", np.array([_count]), np.array([scores[0]]), "knn scores 0")

    return scores




def reshape_data(data:np.array):
    '''把数据除了第一维以外全部压缩成一维, 无转置, 返回bs * dim'''
    shape = data.shape
    data = data.reshape(shape[0], -1) 
    return data

def cross_validation(data, cv=5, rule='scott'):
    if rule =='scott':
        kde = KernelDensity(
        bandwidth='scott',
        kernel='gaussian'
        )
        kde.fit(data)
        return kde
    
    elif rule == "cross_validation":
        # 扩大带宽搜索范围，使用更大的基准带宽
        bandwidths = np.logspace(-1, 2, 30)  # 改为-1到2，增加搜索点数
        
        # 使用Scott's Rule作为参考带宽
        n_samples = data.shape[0]
        n_dims = data.shape[1]
        scott_bandwidth = n_samples ** (-1. / (n_dims + 4))
        
        # 在Scott's Rule周围额外添加一些搜索点
        extra_bandwidths = np.linspace(scott_bandwidth * 0.1, scott_bandwidth * 10, 10)
        bandwidths = np.unique(np.concatenate([bandwidths, extra_bandwidths]))
        
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            cv=cv
        )
        grid.fit(data)
        print(f"Scott's Rule建议带宽: {scott_bandwidth:.3f}")
        print(f"最佳带宽参数: {grid.best_params_['bandwidth']:.3f}")
        print(f"最佳得分 (log-likelihood): {grid.best_score_:.3f}")
        
        kde = KernelDensity(
            bandwidth=grid.best_params_['bandwidth'],
            # bandwidth=scott_bandwidth,
            kernel='gaussian'
        )
        kde.fit(data)
        return kde

    else:
        raise ValueError("rule not supported")

def calculate_scores_kde(states, memory, encoder=None, band_witdth=None, k=10, dist_score=ranked_avg_knn_scores,
                     knn=batch_count_scaled_knn, plotter = None, _count = None):    #后三个参数保持接口一致，位置参数是必须传这个参数，关键字是可以调换参数，也可以没有，必须在后面
    """
    Calculating KDE scores for each of the states. 
    We want to
    optionally encode ALL the states in the buffer to calculate things.

    Parameters
    ----------
    states: States to calculate scores from of size [batch_size x (state_dim)]
    encoder: Encoder that takes in [batch_size x (state_dim)] and returns [batch_size x encoded_size]
    batch_size: batch size of the states ie.[62, 4]
    state_dim: state dimension ie. [32, 32]

    Returns
    -------
    Scores of size [batch_size]
    """
    n_sample = 100 #random sample k point to test average rho
    
    # don't calculate gradients here!
    with torch.no_grad():
        if encoder is None:
            encoder = lambda x: x

        # one big bad batch
        encoded_memory = encoder(torch.tensor(memory, dtype=torch.float).to(device))
        #small_batch 32
        # encoded_memory = calculate_large_batch(encoder, memory)

        if encoded_memory.shape[0] < np.prod(encoded_memory.shape[1:-1]):
            raise ValueError("!!!!!!!!!!!!!!!Encoded memory batch size too small!!!!!!!!!!!!!")

        encoded_states = states
        # REFACTOR THIS
        if encoded_states.shape[-1] != encoded_memory.shape[-1]:
            # encoded_states = encoder(torch.tensor(states, dtype=torch.float).to(device))
            encoded_states = calculate_large_batch(encoder, states)
        

        
        #reshape data
        reshape_states = encoded_states.cpu().detach().numpy() if not isinstance(encoded_states, np.ndarray) else encoded_states
        reshape_memory = encoded_memory.cpu().detach().numpy() if not isinstance(encoded_memory, np.ndarray) else encoded_memory
        
        reshape_states = reshape_data(reshape_states)
        reshape_memory = reshape_data(reshape_memory)
        
        #scale data
        mean_memory = np.mean(reshape_memory, axis=0).reshape(1, -1)
        std_memory = np.std(reshape_memory, axis=0).reshape(1, -1)
        std_memory = np.maximum(std_memory, 1e-6)
        scale_memory = (reshape_memory - mean_memory) / std_memory
        scale_states = (reshape_states - mean_memory) / std_memory
        # scale_random = (random_points - mean_memory) / std_memory
        
        #calculate 3 kde
        kde = cross_validation(scale_memory, cv=5, rule='scott')
        states_rho = kde.score_samples(scale_states) #log(rho) bs
        memory_rho = kde.score_samples(scale_memory) #log(rho) bs
        avg_rho = np.mean(memory_rho)
        std_rho = np.std(memory_rho)
        rewards = -(states_rho - avg_rho) / np.maximum(std_rho, 1e-6) #bs
        
        # print("avg_rho", avg_rho)
        # print("std_rho", std_rho)
        # print("rewards", rewards)
        # print("states_rho", states_rho)
        # print("memory_rho", memory_rho)
        

        # self._plotter.plot("intrinsic_mean_rewards", np.array([self._count]), [np.mean(intr_rewards)], title_name="Intrinsic Rewards")
        # plotter.plot("result0", np.array([_count]), np.array([result[0]]), title_name="result0")
        # plotter.plot("density newest", np.array([_count]), np.array([result[-1]]), title_name="density newest")

    return rewards


def calculate_scores_pre_error(states, memory, encoder=None, band_witdth=None, k=10, dist_score=ranked_avg_knn_scores,
                     knn=batch_count_scaled_knn, plotter = None, _count = None, nsrs : NSRS= None):    #后三个参数保持接口一致，位置参数是必须传这个参数，关键字是可以调换参数，也可以没有，必须在后面
    """
    Calculating KDE scores for each of the states. 
    We want to
    optionally encode ALL the states in the buffer to calculate things.

    Parameters
    ----------
    states: States to calculate scores from of size [batch_size x (state_dim)]
    encoder: Encoder that takes in [batch_size x (state_dim)] and returns [batch_size x encoded_size]
    batch_size: batch size of the states ie.[62, 4]
    state_dim: state dimension ie. [32, 32]

    Returns
    -------
    Scores of size [batch_size]
    """
    if nsrs is None:
        raise ValueError("nsrs is None")

    # don't calculate gradients here!
    with torch.no_grad():
        if encoder is None:
            encoder = lambda x: x

        # one big bad batch
        encoded_memory = encoder(torch.tensor(memory, dtype=torch.float).to(device))
        #small_batch 32
        # encoded_memory = calculate_large_batch(encoder, memory)

        if encoded_memory.shape[0] < np.prod(encoded_memory.shape[1:-1]):
            raise ValueError("!!!!!!!!!!!!!!!Encoded memory batch size too small!!!!!!!!!!!!!")

        encoded_states = states
        # REFACTOR THIS
        if encoded_states.shape[-1] != encoded_memory.shape[-1]:
            # encoded_states = encoder(torch.tensor(states, dtype=torch.float).to(device))
            encoded_states = calculate_large_batch(encoder, states)
        

        
        #reshape data
        reshape_states = encoded_states.cpu().detach().numpy() if not isinstance(encoded_states, np.ndarray) else encoded_states
        reshape_memory = encoded_memory.cpu().detach().numpy() if not isinstance(encoded_memory, np.ndarray) else encoded_memory
        
        reshape_states = reshape_data(reshape_states)
        reshape_memory = reshape_data(reshape_memory)
        
        # ?
        nsrs.calc_nstep_transition_loss()
        

        # self._plotter.plot("intrinsic_mean_rewards", np.array([self._count]), [np.mean(intr_rewards)], title_name="Intrinsic Rewards")
        # plotter.plot("result0", np.array([_count]), np.array([result[0]]), title_name="result0")
        # plotter.plot("density newest", np.array([_count]), np.array([result[-1]]), title_name="density newest")

    return rewards

if __name__ == '__main__':
    batch_size = [5,1]
    query_size = [3, 1]
    data_dim = [2]
    
    # data = np.array([[0,0], [1,1], [1,-1],[-1,1],[-1,-1]])
    # query = np.array([[-1, 0], [0, 1], [-2, -1]])
    data = np.random.rand(13,4,3,3)
    query = np.random.rand(4,4,3,3)
    
    
    
    data = torch.tensor(data).to('cuda')
    query = torch.tensor(query).to('cuda')
    calculate_scores_kde(query, data, encoder=None, band_witdth=None)