"""
Exploration helpers. Requires PyTorch
"""
import torch
from nsrl.helper.knn import ranked_avg_knn_scores, avg_knn_scores, batch_knn, batch_count_scaled_knn
from nsrl.helper.pytorch import device, calculate_large_batch


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
                     knn=batch_count_scaled_knn):
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

    return scores