"""
Novelty Search in Representational Space (NSRS) algorithm.
"""

import copy
import numpy as np

from ..base_classes import LearningAlgo
from .NN_pytorch import NN # Default Neural network used
from nsrl.helper.knn import ranked_avg_knn_scores, avg_knn_scores, batch_knn, batch_count_scaled_knn
from nsrl.helper.pytorch import device

import torch
import torch.nn.functional as F
import torch.optim as optim

def mean_squared_error_p_pytorch(y_pred, target=1.0):
    """ Modified mean square error that clips
    """
    return  torch.sum(torch.clamp( (torch.mean((y_pred)**2,dim=-1)[0] - target), 0., 100.)) # = modified mse error L_inf

def exp_dec_error_pytorch(y_pred):
    return torch.mean(torch.exp( - 5.*torch.sqrt( torch.clamp(torch.sum(y_pred**2, dim=-1),0.000001,10) )  ))
    # return torch.mean(torch.exp( - 2.5*torch.sqrt( torch.clamp(torch.sum(y_pred**2, dim=-1),0.000001,10) )  ))

def exp_dec_error_pytorch_2(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean()

def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

def cosine_proximity2_pytorch(y_true, y_pred):
    """ This loss is similar to the native cosine_proximity loss from Keras
    but it differs by the fact that only the two first components of the two vectors are used
    """

    y_true = F.normalize(y_true[:,0:2],p=2,dim=-1)
    y_pred = F.normalize(y_pred[:,0:2],p=2,dim=-1)
    return -torch.sum(y_true * y_pred, dim=-1)

class NSRS(LearningAlgo):
    def __init__(self,
                 environment, rho=0.9, rms_epsilon=0.0001, beta=0.0,
                 momentum=0, clip_norm=0, freeze_interval=100,
                 batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(),
                 neural_network=NN, learn_representation=True, k=10,
                 score_func=ranked_avg_knn_scores, knn=batch_count_scaled_knn,
                 obs_per_state=1, action_type='q_argmax', **kwargs):
        super(NSRS, self).__init__(environment, batch_size)
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_norm = clip_norm
        self._update_rule = update_rule
        self._freeze_interval = freeze_interval
        self._random_state = random_state
        self.q_update_counter = 0
        self.repr_update_counter = 0
        self._high_dim_obs = kwargs.get('higher_dim_obs', False)
        self._internal_dim = kwargs.get('internal_dim', 2)
        self._predictor_network = kwargs.get('predictor_network', False)
        self._encoder_prop_td = kwargs.get('encoder_prop_td', True)
        self._rnn_q_func = kwargs.get('rnn_q_func', False)
        self._beta = beta
        self._learn_representation = learn_representation
        self._depth = kwargs.get('depth', 1)
        self._train_csc_ent = kwargs.get('train_csc_ent', False)
        self._train_csc_dist = kwargs.get('train_csc_dist', False)
        self._train_linf_dist = kwargs.get('train_linf_dist', False)
        self._consec_dist = kwargs.get('consec_dist', 0.5)
        self._train_reward = kwargs.get('train_reward', False)
        self._score_func = score_func
        self._knn = knn
        self._k = k
        self._double_Q = kwargs.get('double_Q', True)
        self._obs_per_state = obs_per_state

        self._action_type = action_type

        dropout_p = kwargs.get('dropout_p', 0.5)
        transition_hidden_units = kwargs.get('transition_hidden_units', 10)
        scale_transition = kwargs.get('scale_transition', False)

        self.learn_and_plan = neural_network(self._batch_size,
                                             self._input_dimensions,
                                             self._n_actions,
                                             self._random_state,
                                             high_dim_obs=self._high_dim_obs,
                                             internal_dim=self._internal_dim,
                                             transition_dropout_p=dropout_p,
                                             transition_hidden_units=transition_hidden_units,
                                             scale_transition=scale_transition,
                                             bootstrap_q_func=self._action_type == 'bootstrap_q')

        self.encoder = lambda x: x
        if self._learn_representation:
            self.encoder = self.learn_and_plan.encoder_model().to(device)

        self.encoder_diff = self.learn_and_plan.encoder_diff_model

        if self._predictor_network:
            self.random_target_network = self.learn_and_plan.encoder_model()
            self.predictor_network = self.learn_and_plan.encoder_model()

        self.R = self.learn_and_plan.float_model().to(device)

        if self._rnn_q_func:
            self.Q = self.learn_and_plan.RNN_Q_model(self.encoder, use_representation=self._learn_representation).to(device)
        else:
            self.Q = self.learn_and_plan.Q_model(use_representation=self._learn_representation).to(device)

        # self.explore_Q = self.learn_and_plan.Q_model()

        self.gamma = self.learn_and_plan.float_model().to(device)
        self.transition = self.learn_and_plan.transition_model().to(device)

        self.all_models = [self.Q]
        if self._learn_representation:
            self.all_models = [self.encoder,self.R,self.Q,self.gamma,self.transition]

        # used to fit Q value
        self.full_Q = self.learn_and_plan.full_Q_model

        # used to fit rewards
        self.full_R = self.learn_and_plan.full_float_model

        # used to fit gamma
        self.full_gamma = self.learn_and_plan.full_float_model

        # used to fit transitions
        self.diff_Tx_x_ = self.learn_and_plan.diff_Tx_x_

        # constraint on consecutive t
        self.diff_s_s_ = self.learn_and_plan.encoder_diff_model

        # Compile all models
        self._compile()

        # Instantiate the same neural network as a target network.
        self.learn_and_plan_target = neural_network(self._batch_size,
                                                    self._input_dimensions,
                                                    self._n_actions,
                                                    self._random_state,
                                                    high_dim_obs=self._high_dim_obs,
                                                    internal_dim=self._internal_dim,
                                                    transition_hidden_units=transition_hidden_units,
                                                    scale_transition=scale_transition,
                                                    bootstrap_q_func=self._action_type == 'bootstrap_q')
        self.encoder_target = lambda x: x
        if self._learn_representation:
            self.encoder_target = self.learn_and_plan_target.encoder_model().to(device)

        if self._rnn_q_func:
            self.Q_target = self.learn_and_plan_target.RNN_Q_model(self.encoder, use_representation=self._learn_representation).to(device)
        else:
            self.Q_target = self.learn_and_plan_target.Q_model(use_representation=self._learn_representation).to(device)

        self.R_target = self.learn_and_plan_target.float_model().to(device)
        self.gamma_target = self.learn_and_plan_target.float_model().to(device)
        self.transition_target = self.learn_and_plan_target.transition_model().to(device)

        self.full_Q_target = self.learn_and_plan_target.full_Q_model

        self.all_models_target = [self.Q_target]
        if self._learn_representation:
            self.all_models_target = [self.encoder_target,self.R_target,self.Q_target,self.gamma_target,self.transition_target]

    @property
    def internal_dim(self):
        return self._internal_dim

    def getAllParams(self):
        """ Provides all parameters used by the learning algorithm

         Returns
         -------
         Values of the parameters: list of state_dicts
         """

        return {'models': [m.state_dict() for m in self.all_models],
                'optimizers': [o.state_dict() for o in self.optimizers]}

    def setAllParams(self, loaded):
        """ Set all parameters used by the learning algorithm

        Arguments
        ---------
        list_of_values : list of numpy arrays
             list of state_dicts to be set (same order than given by getAllParams()).
        """
        list_of_values = loaded['models']
        for i, m in enumerate(self.all_models):
            m.load_state_dict(list_of_values[i])

        self._resetQHat()

        list_of_optims = loaded['optimizers']
        for i, o in enumerate(self.optimizers):
            o.load_state_dict(list_of_optims[i])

    def _compile(self):
        """ Compile all the optimizers for the different losses
        """

        if (self._update_rule == "rmsprop"):
            self.optimizer_full_Q = optim.RMSprop(self.Q.parameters(), lr=self._lr, alpha=self._rho,
                                                  eps=self._rms_epsilon)
            self.optimizer_diff_Tx_x_ = None
            self.optimizer_full_R = None
            self.optimizer_full_gamma = None
            self.optimizer_encoder = None
            self.optimizer_encoder_diff = None
            self.optimizer_diff_s_s_ = None

            if self._learn_representation:
                q_params = list(self.Q.parameters())
                if self._encoder_prop_td:
                    q_params = list(self.encoder.parameters()) + q_params
                self.optimizer_full_Q = optim.RMSprop(q_params,
                                                      lr=self._lr, alpha=self._rho, eps=self._rms_epsilon)
                self.optimizer_diff_Tx_x_ = optim.RMSprop(
                    list(self.encoder.parameters()) + list(self.transition.parameters()), lr=self._lr, alpha=self._rho,
                    eps=self._rms_epsilon)  # Different optimizers for each network;
                self.optimizer_full_R = optim.RMSprop(list(self.encoder.parameters()) + list(self.R.parameters()),
                                                      lr=self._lr, alpha=self._rho,
                                                      eps=self._rms_epsilon)  # to possibly modify them separately
                self.optimizer_full_gamma = optim.RMSprop(list(self.encoder.parameters()) + list(self.gamma.parameters()),
                                                          lr=self._lr, alpha=self._rho, eps=self._rms_epsilon)
                self.optimizer_encoder = optim.RMSprop(self.encoder.parameters(), lr=self._lr, alpha=self._rho,
                                                       eps=self._rms_epsilon)
                self.optimizer_encoder_diff = optim.RMSprop(self.encoder.parameters(), lr=self._lr, alpha=self._rho,
                                                            eps=self._rms_epsilon)
                self.optimizer_diff_s_s_ = optim.RMSprop(self.encoder.parameters(), lr=self._lr, alpha=self._rho,
                                                         eps=self._rms_epsilon)

                self.optimizer_repr = optim.RMSprop(list(self.encoder.parameters()) + list(self.transition.parameters()) + list(self.R.parameters()),
                                                    lr=self._lr, alpha=self._rho, eps=self._rms_epsilon)

            # self.optimizer_force_features=optim.RMSprop(list(self.encoder.parameters()) + list(self.transition.parameters()), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon) # This never gets updated

        else:
            raise Exception('The update_rule ' + self._update_rule + ' is not implemented.')

        self.optimizers = [self.optimizer_full_Q, self.optimizer_diff_Tx_x_,
                           self.optimizer_full_R, self.optimizer_full_gamma,
                           self.optimizer_encoder, self.optimizer_encoder_diff,
                           self.optimizer_diff_s_s_]
        self.optimizers = [optim for optim in self.optimizers if optim is not None]

    def calc_nstep_transition_loss(self, abstr_states, nstep_states, nstep_onehot_actions, nstep_terminals,
                                   validation=False, normalize=False):

        steps = nstep_onehot_actions.shape[1]
        # initial abstract states

        loss_val = torch.tensor(0.0, requires_grad=False, device=device, dtype=torch.float)
        validation_tensors = torch.zeros(abstr_states.shape[0]).to(device)

        for i in range(steps):
            # get onehot actions for step i
            action = nstep_onehot_actions[:, i]

            # append to abstr states as transition_model inputs
            transition_inputs = torch.cat((abstr_states, action), dim=-1)
            prev_abstr_states = abstr_states
            abstr_states = self.transition(transition_inputs)

            # get our target abstract states at step i + 1
            current_state = nstep_states[:, i + 1:self._obs_per_state + i + 1]
            if self._obs_per_state == 1:
                current_state = current_state.squeeze(1)
            target_abstr_states = self.encoder(current_state)

            def lalign(x, y, alpha=2):
                return (x - y).norm(dim=1).pow(alpha)

            # find diff and mask with terminals from step i
            # diff = torch.sum(((abstr_states - target_abstr_states) * (1 - nstep_terminals[:, i])).norm(dim=-1).pow(2), dim=-1)
            lv = lalign(abstr_states, target_abstr_states)
            if normalize:
                state_diff = F.pairwise_distance(prev_abstr_states, abstr_states, p=2.0).detach()
                lv /= state_diff

            if validation:
                # if we calculate validation values, sum diff over dims of abstract states and square.
                val_diff = ((abstr_states - target_abstr_states) * (1 - nstep_terminals[:, i])).norm(dim=-1).pow(2)
                validation_tensors += val_diff
            # loss_val += loss_func(diff, torch.zeros_like(diff))
            loss_val += lv.mean()

        # Normalization
        validation_tensors /= steps
        loss_val /= steps

        if validation:
            return loss_val, validation_tensors

        return loss_val, abstr_states

    def train_repr(self, nstep_states, nstep_actions, nstep_rewards, nstep_terminals, training=True, scale=1):
        """
        Train representations from one batch of data. This should be run multiple steps
        per "training phase". agent.run() should alternate between this and
        exploration using the abstract representations its learnt.

        Parameters
        ----------
        states: [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        actions: [self._batch_size]
        rewards: [self._batch_size]
        nextStates: [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        terminals: [self._batch_size]

        Returns
        -------
        """
        # Start training mode
        for m in self.all_models: m.train()

        nstep_onehot_actions = np.zeros((nstep_actions.shape[0], nstep_actions.shape[1], self._n_actions))
        actions_idx = nstep_actions[:, :, None]
        first_axis_idx = np.tile(np.arange(nstep_onehot_actions.shape[0]), (actions_idx.shape[1], 1)).transpose()[:, :, None]
        second_axis_idx = np.tile(np.arange(nstep_onehot_actions.shape[1]), (actions_idx.shape[0], 1))[:, :, None]

        nstep_onehot_actions[first_axis_idx, second_axis_idx, actions_idx] = 1
        onehot_actions = nstep_onehot_actions[:, 0]

        nstep_onehot_actions = torch.from_numpy(nstep_onehot_actions)
        onehot_actions = torch.from_numpy(onehot_actions)


        if isinstance(nstep_states, np.ndarray):
            nstep_states = torch.from_numpy(nstep_states)

        states = nstep_states[:, 0:self._obs_per_state]
        next_states = nstep_states[:, 1:self._obs_per_state + 1]
        if self._obs_per_state == 1:
            states = states.squeeze(1)
            next_states = next_states.squeeze(1)

        if isinstance(nstep_terminals, np.ndarray):
            nstep_terminals = torch.from_numpy(nstep_terminals)
        nstep_terminals = nstep_terminals.unsqueeze(-1)
        terminals = nstep_terminals[:, 0].float()

        if isinstance(nstep_rewards, np.ndarray):
            nstep_rewards = torch.from_numpy(nstep_rewards)
        nstep_rewards = nstep_rewards.unsqueeze(-1)
        rewards = nstep_rewards[:, 0]

        states, nstep_states, onehot_actions, nstep_onehot_actions, \
        next_states, terminals, nstep_terminals, rewards, nstep_rewards = \
            [t.float().to(device) for t in
             [states, nstep_states, onehot_actions, nstep_onehot_actions,
              next_states, terminals, nstep_terminals,
              rewards, nstep_rewards]]

        losses = {}
        all_loss_vals = torch.tensor(0).to(device).float()

        abstr_state = self.encoder(states)
        next_abstr_state = self.encoder(next_states)

        # REWARD LOSS
        if self._train_reward:
            reward_pred = self.R(torch.cat((abstr_state, onehot_actions), dim=1))
            loss = torch.nn.MSELoss()
            loss_val = loss(reward_pred, rewards)
            all_loss_vals += loss_val
            losses['reward_loss'] = loss_val.item()


        # now on gamma
        # if training:
        #     self.optimizer_full_gamma.zero_grad()
        # out = self.full_gamma(states,onehot_actions,self.encoder,self.gamma)
        # loss = torch.nn.MSELoss()
        # loss_val = loss(out,(1-terminals[:])*self._df)
        # all_loss_vals += loss_val
        # losses['gamma_loss'] = loss_val.item()


        # We have our transition loss.
        loss_val, transition_states = self.calc_nstep_transition_loss(abstr_state, nstep_states, nstep_onehot_actions, nstep_terminals)
        all_loss_vals += loss_val

        losses['transition_loss'] = loss_val.item()

        # This one is very important
        # Entropy maximization loss (through exponential) between two random states
        # this loss is (indirectly) enforcing the radius 1 condition

        # loss_val = (exp_dec_error_pytorch_2(abstr_state) + exp_dec_error_pytorch_2(next_abstr_state)) / 2
        loss_val = (lunif(abstr_state) + lunif(next_abstr_state)) / 2
        all_loss_vals += loss_val
        losses['two_random_state_entropy_max_loss'] = loss_val.item()


        # Entropy maximization loss (through exponential) between two consecutive states
        if self._train_csc_ent:
            out = self.diff_s_s_(self.encoder,states,next_states)
            # loss_val += exp_dec_error_pytorch(out) * self._beta
            loss_val = exp_dec_error_pytorch(out) * self._beta
            all_loss_vals += loss_val
            losses['two_consecutive_state_entropy_max_loss'] = loss_val.item()


        # Consec dist 1 loss
        if self._train_csc_dist:
            out = abstr_state - next_abstr_state
            csc_dist_1_loss = mean_squared_error_p_pytorch(out, target=self._consec_dist)
            loss_val = csc_dist_1_loss
            all_loss_vals += loss_val
            losses['consecutive_dist_1_loss'] = csc_dist_1_loss.item()

        if self._train_linf_dist:
            loss_val = mean_squared_error_p_pytorch(abstr_state, target=1)
            all_loss_vals += loss_val
            losses['L_inf_radius_1'] = loss_val.item()


        if training:
            # We optimize here a bit - one backward pass for all our optimizers.
            if self._train_reward:
                self.optimizer_full_R.zero_grad()
            self.optimizer_encoder.zero_grad()
            self.optimizer_diff_Tx_x_.zero_grad()

            all_loss_vals.backward()
            # for param in list(self.encoder.parameters()) + list(self.transition.parameters()) + list(self.R.parameters()):
            #     param.grad.data.clamp_(-1, 1)

            if self._train_reward:
                self.optimizer_full_R.step()

            self.optimizer_encoder.step()
            self.optimizer_diff_Tx_x_.step()

        if(self.repr_update_counter%1000==0):
            print ("Number of training repr steps:"+str(self.repr_update_counter)+".")

        self.repr_update_counter += 1

        # HERE
        with torch.no_grad():
            for m in self.all_models: m.eval()

            test_abstr_state = self.encoder(states)
            test_abstr_next_state = self.encoder(next_states)
            normalize_losses, transition_loss_ind = self.calc_nstep_transition_loss(
                test_abstr_state, nstep_states, nstep_onehot_actions, nstep_terminals, validation=True, normalize=True)
            losses['inference_normalized_transition_losses'] = normalize_losses.item()
            random_states_loss_ind = (lunif(test_abstr_state) + lunif(test_abstr_next_state)) / 2

        return losses, random_states_loss_ind.cpu().numpy(), transition_loss_ind.cpu().numpy()

    def bootstrap_q_update(self, nstep_states, nstep_actions, nstep_rewards, nstep_terminals, nstep_masks):
        states = nstep_states[:, :self._obs_per_state]
        if states.shape[1] == 1:
            states = states.squeeze(1)
        next_states = nstep_states[:, -self._obs_per_state:]
        if next_states.shape[1] == 1:
            next_states = next_states.squeeze(1)
        actions = nstep_actions[:, 0]
        rewards = nstep_rewards[:, 0]
        terminals = nstep_terminals[:, 0]
        masks = nstep_masks[:, 0]
        head_num = masks.shape[-1]

        mse_losses = []
        abstr_states = self.encoder(states)
        q_pred_all = self.Q(abstr_states, list(range(head_num)))
        next_q_vals_current_qnet_all = None
        if self._double_Q:
            next_q_vals_current_qnet_all = self.Q(abstr_states, list(range(head_num)))

        abstr_next_states = self.encoder_target(next_states)
        next_q_pred_all = self.Q_target(abstr_next_states, list(range(head_num)))

        for i in range(head_num):
            q_pred = q_pred_all[i]
            q_s_a = q_pred.gather(1, actions.long())
            next_q_pred = next_q_pred_all[i]
            if self._double_Q:
                next_q_vals_current_qnet = next_q_vals_current_qnet_all[i]
                argmax_next_q_vals = next_q_vals_current_qnet.argmax(dim=-1).unsqueeze(-1)
                max_next_q_pred = torch.gather(next_q_pred, 1, argmax_next_q_vals)
            else:
                max_next_q_pred = next_q_pred.max(1, keepdim=True)[0]
            target_q_s_a = rewards + self._df * (1 - terminals) * max_next_q_pred

            mse_loss = (q_s_a - target_q_s_a) ** 2
            mse_losses.append(mse_loss)

        mse_losses = torch.cat(mse_losses, dim=1)
        qf_loss = (mse_losses * masks / head_num).sum(1).mean()

        self.optimizer_full_Q.zero_grad()
        qf_loss.backward()
        self.optimizer_full_Q.step()

        self.q_update_counter += 1

        return qf_loss, q_pred_all

    def train_q(self, nstep_states, nstep_actions, nstep_rewards, nstep_terminals, nstep_masks=None):
        """
        Train Q value approximator from one batch of data. This should be run multiple steps
        per "training phase".

        Parameters
        ----------
        states: [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        actions: [batch_size]
        rewards: [batch_size]
        next_states: [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        terminals: [batch_size]

        Returns
        -------
        """

        # Start training mode
        for m in self.all_models: m.train()

        if isinstance(nstep_actions, np.ndarray):
            nstep_actions = torch.from_numpy(nstep_actions.astype(int)[:, None])
        if isinstance(nstep_states, np.ndarray):
            nstep_states = torch.from_numpy(nstep_states)

        if isinstance(nstep_terminals, np.ndarray):
            nstep_terminals = torch.from_numpy(nstep_terminals).unsqueeze(-1)
        if isinstance(nstep_rewards, np.ndarray):
            nstep_rewards = torch.from_numpy(nstep_rewards).unsqueeze(-1)

        if nstep_masks is not None and isinstance(nstep_masks, np.ndarray):
            nstep_masks = torch.from_numpy(nstep_masks)
            nstep_masks = nstep_masks.float().to(device)

        nstep_states, nstep_actions, nstep_terminals, nstep_rewards = \
            [t.float().to(device) for t in [nstep_states, nstep_actions, nstep_terminals, nstep_rewards]]
        nstep_actions = nstep_actions.long()


        # Q Learning loss (TD lambda)
        if self.q_update_counter % self._freeze_interval == 0:
            print("updating targets")
            self._resetQHat()

        if nstep_masks is not None:
            # CURRENTLY WORKS ONLY FOR N = 1
            return self.bootstrap_q_update(nstep_states, nstep_actions, nstep_rewards, nstep_terminals, nstep_masks)
        steps = nstep_rewards.shape[1]

        with torch.no_grad():
            final_state = nstep_states[:, -self._obs_per_state:]
            if self._obs_per_state == 1:
                final_state = final_state.squeeze(-1)

            final_q_vals = self.full_Q_target(final_state, self.encoder_target, self.Q_target).detach()
            if self._double_Q:
                next_q_vals_current_qnet = self.full_Q(final_state, self.encoder, self.Q)
                argmax_next_q_vals = next_q_vals_current_qnet.argmax(dim=-1).unsqueeze(-1)
                max_final_q_vals = torch.gather(final_q_vals, 1, argmax_next_q_vals).squeeze(-1)
            else:
                max_final_q_vals, argmax_final_q_vals = torch.max(final_q_vals, dim=-1)
        not_terminals = (1 - nstep_terminals[:, -1])
        next_q_vals = ((self._df ** steps) * max_final_q_vals).unsqueeze(-1)

        discounts = np.array([self._df ** i for i in range(steps)])[:, None]
        batch_discounts = np.repeat(discounts, nstep_rewards.shape[0], axis=1).T[:, :, None]
        discounted_rewards = np.sum(np.multiply(batch_discounts, nstep_rewards.cpu().numpy()), axis=1) # b x 1
        discounted_rewards = torch.from_numpy(discounted_rewards).float().to(device)
        target = discounted_rewards + not_terminals * next_q_vals

        self.optimizer_full_Q.zero_grad()
        initial_states = nstep_states[:, :self._obs_per_state]
        if self._obs_per_state == 1:
            initial_states = initial_states.squeeze(1)

        q_vals_all = self.full_Q(initial_states, self.encoder, self.Q, static_encoder=not self._encoder_prop_td)

        # calculate Q values for each calculated reward:
        q_vals = q_vals_all.gather(1, nstep_actions[:, 0])
        q_loss_val = F.smooth_l1_loss(q_vals, target)

        q_loss_val.backward()
        params = list(self.Q.parameters())

        if self._learn_representation and self._encoder_prop_td:
            params = list(self.encoder.parameters()) + params

        for param in params:
            param.grad.data.clamp_(-1, 1)
        self.optimizer_full_Q.step()

        if(self.q_update_counter % 1000 ==0):
            print ("Number of training Q steps:"+str(self.q_update_counter)+".")

        self.q_update_counter += 1

        return q_loss_val, (q_vals.detach()-target)**2


    def qValues(self, state_val):
        """ Get the q values for one pseudo-state (without planning)

        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).

        Returns
        -------
        The q values for the provided pseudo state
        """
        copy_state=copy.deepcopy(state_val) #Required!
        state_tensor = torch.tensor(copy_state, dtype=torch.float).to(device)
        return self.full_Q(state_tensor, self.encoder, self.Q)

    def qValues_planning(self, state_val, R, gamma, T, Q, d=5):
        """ Get the average Q-values up to planning depth d for one pseudo-state.

        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).
        R : float_model
            Model that fits the reward
        gamma : float_model
            Model that fits the discount factor
        T : transition_model
            Model that fits the transition between abstract representation
        Q : Q_model
            Model that fits the optimal Q-value
        d : int
            planning depth

        Returns
        -------
        The average q values with planning depth up to d for the provided pseudo-state
        """
        encoded_x = self.encoder.predict(state_val)

        #        ## DEBUG PURPOSES
        #        print ( "self.full_Q.predict(state_val)[0]" )
        #        print ( self.full_Q.predict(state_val)[0] )
        #        identity_matrix = np.diag(np.ones(self._n_actions))
        #        if(encoded_x.ndim==2):
        #            tile3_encoded_x=np.tile(encoded_x,(self._n_actions,1))
        #        elif(encoded_x.ndim==4):
        #            tile3_encoded_x=np.tile(encoded_x,(self._n_actions,1,1,1))
        #        else:
        #            print ("error")
        #
        #        repeat_identity=np.repeat(identity_matrix,len(encoded_x),axis=0)
        #        ##print tile3_encoded_x
        #        ##print repeat_identity
        #        r_vals_d0=np.array(R.predict([tile3_encoded_x,repeat_identity]))
        #        #print "r_vals_d0"
        #        #print r_vals_d0
        #        r_vals_d0=r_vals_d0.flatten()
        #        print "r_vals_d0"
        #        print r_vals_d0
        #        next_x_predicted=T.predict([tile3_encoded_x,repeat_identity])
        #        #print "next_x_predicted"
        #        #print next_x_predicted
        #        one_hot_first_action=np.zeros((1,self._n_actions))
        #        one_hot_first_action[0]=1
        #        next_x_predicted=T.predict([next_x_predicted[0:1],one_hot_first_action])
        #        next_x_predicted=T.predict([next_x_predicted[0:1],one_hot_first_action])
        #        next_x_predicted=T.predict([next_x_predicted[0:1],one_hot_first_action])
        #        #print "next_x_predicted action 0 t4"
        #        #print next_x_predicted
        #        ## END DEBUG PURPOSES

        QD_plan = 0
        for i in range(d + 1):
            Qd = self.qValues_planning_abstr(encoded_x, R, gamma, T, Q, d=i,
                                             branching_factor=[self._n_actions, 2, 2, 2, 2, 2, 2, 2]).reshape(
                len(encoded_x), -1)
            QD_plan += Qd
        QD_plan = QD_plan / (d + 1)

        return QD_plan

    def qValues_planning_abstr(self, state_abstr_val, R, gamma, T, Q, d, branching_factor=None):
        """ Get the q values for pseudo-state(s) with a planning depth d.
        This function is called recursively by decreasing the depth d at every step.

        Arguments
        ---------
        state_abstr_val : internal state(s).
        R : float_model
            Model that fits the reward
        gamma : float_model
            Model that fits the discount factor
        T : transition_model
            Model that fits the transition between abstract representation
        Q : Q_model
            Model that fits the optimal Q-value
        d : int
            planning depth

        Returns
        -------
        The Q-values with planning depth d for the provided encoded state(s)
        """
        # if(branching_factor==None or branching_factor>self._n_actions):
        #    branching_factor=self._n_actions

        n = len(state_abstr_val)
        identity_matrix = np.identity(self._n_actions)

        # THIS POP SEEMS SUSPICIOUS - check this along with line 557
        this_branching_factor = branching_factor.pop(0)
        if (n == 1):
            # We require that the first branching factor is self._n_actions so that this function return values
            # with the right dimension (=self._n_actions).
            this_branching_factor = self._n_actions

        if (d == 0):
            if (this_branching_factor < self._n_actions):
                # HERE
                return np.partition(Q.predict(state_abstr_val).cpu().detach().numpy(), -this_branching_factor)[:,
                       -this_branching_factor:]
            else:
                return Q.predict(state_abstr_val).cpu().detach().numpy()  # no change in the order of the actions
        else:
            if (this_branching_factor == self._n_actions):
                # All actions are considered in the tree
                # NB: For this case, we do not use argpartition because we want to keep the actions in the natural order
                # That way, this function returns the Q-values for all actions with planning depth d in the right order
                repeat_identity = np.repeat(identity_matrix, len(state_abstr_val), axis=0)
                if (len(state_abstr_val.shape) == 2):
                    tile3_encoded_x = np.tile(state_abstr_val.cpu().detach().numpy(), (self._n_actions, 1))
                elif (len(state_abstr_val.shape) == 4):
                    tile3_encoded_x = np.tile(state_abstr_val.cpu().detach().numpy(), (self._n_actions, 1, 1, 1))
                else:
                    print("error")
            else:
                # A subet of the actions corresponding to the best estimated Q-values are considered et each branch
                # HERE
                estim_Q_values = Q.predict(state_abstr_val)
                ind = np.argpartition(estim_Q_values.cpu().detach().numpy(), -this_branching_factor)[:,
                      -this_branching_factor:]
                # Replacing ind if we want random branching
                # ind = np.random.randint(0,self._n_actions,size=ind.shape)
                repeat_identity = identity_matrix[ind].reshape(n * this_branching_factor, self._n_actions)
                tile3_encoded_x = np.repeat(state_abstr_val.cpu().detach().numpy(), this_branching_factor, axis=0)

            float_model_input_np = np.concatenate([tile3_encoded_x, repeat_identity], axis=-1)
            input_tensor = torch.tensor(float_model_input_np, dtype=torch.float).to(device)
            r_vals_d0 = R.predict(input_tensor).cpu().detach().numpy()
            r_vals_d0 = r_vals_d0.flatten()

            gamma_vals_d0 = gamma.predict(input_tensor).cpu().detach().numpy()
            gamma_vals_d0 = gamma_vals_d0.flatten()

            next_x_predicted = T.predict(input_tensor)
            predicted = self.qValues_planning_abstr(next_x_predicted, R, gamma, T, Q, d=d - 1,
                                            branching_factor=branching_factor)
            predicted = predicted.reshape(
                    len(state_abstr_val) * this_branching_factor, branching_factor[0])
            return r_vals_d0 + gamma_vals_d0 * np.amax(predicted, axis=1).flatten()

    def intrRewards_planning(self, abstr_state, transition, all_prev_states, R=None, ret_transitions=False):
        num_actions = self.learn_and_plan.n_actions

        # indices of actions
        actions_idx = torch.arange(0, num_actions).to(device)

        # one hot encoding of actions
        actions = torch.zeros(num_actions, num_actions).to(device)
        actions[actions_idx, actions_idx] = 1

        # tensor of repeated abstr reps of states
        repeated_states = abstr_state.repeat(num_actions, 1)

        # concatenate abstr rep w/ one-hot actions
        state_actions = torch.cat((repeated_states, actions), dim=-1)

        # calculate states of new states after transitions
        new_states = transition(state_actions)

        # get scores
        scores = self._score_func(new_states.cpu().detach().numpy(),
                                  all_prev_states.cpu().detach().numpy(),
                                  k=self._k, knn=self._knn)
        if R is not None:
            scores += R(state_actions).squeeze(-1).cpu().detach().numpy()

        if ret_transitions:
            return scores, new_states

        return scores

    def novelty_one_step_planning(self, abstr_state, Q, transition, all_prev_states, R=None):
        # with torch.no_grad():
        # get intrinsic rewards from next states
        rewards, new_states = \
            self.intrRewards_planning(abstr_state, transition, all_prev_states, R=R, ret_transitions=True)

        # calculate Q values from next states
        q_values_new_states = Q(new_states) # b (n_actions) x n_actions

        # discount maximum and add to intrinsic rewards
        max_q_new_states, argmax_q_new_states = torch.max(q_values_new_states, dim=-1)
        scores = rewards + self._df * max_q_new_states.cpu().detach().numpy()

        return scores

    def chooseBestAction(self, state, mode, *args, **kwargs):
        """
        Simple max-q-value planning.
        We use this for now, will graduate to full on path planning
        once we figure everything out

        Parameters
        ----------
        state
        mode
        args
        kwargs

        Returns
        -------

        """

        copy_state = copy.deepcopy(state)  # Required because of the "hack" below
        state_tensor = torch.tensor(copy_state[0], dtype=torch.float).to(device)

        with torch.no_grad():
            if(mode==None):
                mode=0
            # di=[0,1,3,6]
            di=[3, 1, 3, 6]
            # We use the mode to define the planning depth
            # HERE
            scores = self.qValues_planning(state_tensor,self.R,self.gamma, self.transition, self.Q, d=di[mode])

        return np.argmax(scores, axis=-1).item(), np.max(scores, axis=-1).item()


    def novelty_d_step_planning(self, abstr_state, Q, transition, all_prev_states, R=None, d=3, b=2):
        """
        Wrapper function to qValues_planning_abstr
        """
        all_prev_states = all_prev_states.cpu().detach().numpy()
        class Reward:
            @staticmethod
            def predict(input_tensor):
                # Input tensor is (s, a)
                new_states = transition(input_tensor)

                rewards = self._score_func(new_states.cpu().detach().numpy(),
                                           all_prev_states, k=self._k,
                                           knn=self._knn)
                if R is not None:
                    predicted_rewards = R(input_tensor).squeeze(-1).cpu().detach().numpy()
                    rewards += predicted_rewards
                # rewards = np.clip(rewards, -1, 1)
                # rewards = ranked_avg_knn_scores(new_states.cpu().detach().numpy(), all_prev_states)
                # rewards = avg_knn_scores(new_states.cpu().detach().numpy(), all_prev_states)
                return torch.from_numpy(rewards).to(device)

        class Gamma:
            @staticmethod
            def predict(input_tensor):
                gammas = np.zeros(len(input_tensor)) + self._df
                return torch.from_numpy(gammas).to(device)

        branching_factors = [self._n_actions] + [b] * (d + 1)
        return self.qValues_planning_abstr(abstr_state, Reward, Gamma, transition, Q, d, branching_factor=branching_factors)


    def setLearningRate(self, lr):
        """ Setting the learning rate

        Parameters
        -----------
        lr : float
            The learning rate that has to be set
        """

        self._lr = lr
        print("New learning rate set to " + str(self._lr) + ".")
        for i, optim in enumerate(self.optimizers):
            for param_group in optim.param_groups:
                param_group['lr'] = lr if i != len(self.optimizers) - 1 else lr / 5.

    def _resetQHat(self):
        """ Set all target weights equal to the main model weights
        """

        for mod,mod_t in zip(self.all_models,self.all_models_target):
            mod_t.load_state_dict(mod.state_dict())
            mod_t.eval()
