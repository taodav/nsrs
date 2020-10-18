
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from nsrl.helper.pytorch import device
import nsrl.learning_algos.inits as inits

class NN():
    """

    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    high_dim_obs : Boolean
        Whether the input is high dimensional (ie. video) or low dimensional (ie. vector)
        be low-dimensional
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state, **kwargs):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._high_dim_obs=kwargs["high_dim_obs"]
        self.internal_dim = kwargs["internal_dim"]  # 2 for laby
        self._transition_dropout_p = kwargs.get("transition_dropout_p", 0.5)
        self._transition_hidden_units = kwargs.get("transition_hidden_units", 10)
        self._scale_transition = kwargs.get('scale_transition', False)
        self._bootstrap_q_func = kwargs.get('bootstrap_q_func', False)


    def encoder_model(self):
        """ Instantiate a PyTorch model for the encoder of the learning algorithm.
        
        The model takes the following as input 
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        
        Parameters
        -----------
        
    
        Returns
        -------
        Keras model with output x (= encoding of s)
    
        """

        # if input is a vector
        self._pooling_encoder=1
        class VectorEncoder(nn.Module):
            def __init__(self,internal_dim,input_dim):
                super(VectorEncoder, self).__init__()
                self.input_dim_flat = np.prod(input_dim)
                self.lin1 = nn.Linear(self.input_dim_flat, 200)
                self.lin2 = nn.Linear(200, 100)
                self.lin3 = nn.Linear(100, 50)
                self.lin4 = nn.Linear(50, 10)
                self.lin5 = nn.Linear(10, internal_dim)


            def forward(self, x):
                x = x.view(-1, self.input_dim_flat)
                x = torch.tanh(self.lin1(x))
                x = torch.tanh(self.lin2(x))
                x = torch.tanh(self.lin3(x))
                x = torch.tanh(self.lin4(x))
                # x = torch.tanh(self.lin5(x))
                x = self.lin5(x)
                # x = torch.norm(self.lin5(x), dim=-1)

                return x

            def predict(self, x):
                if len(x.shape) == 4:
                    # this means we have b x in_channels x h x w
                    x = x.view(x.shape[0], -1)

                return self.forward(x)

        class ResnetEncoder(nn.Module):
            def __init__(self, internal_dim, input_dim, pretrained=True):
                super(ResnetEncoder, self).__init__()
                resnet = torchvision.models.resnet18(pretrained=pretrained)
                self.resnet_head = nn.Sequential(*list(resnet.children())[0:8])
                self.conv = nn.Sequential(*filter(bool, [
                    nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
                    nn.ReLU()
                ]))
                # convolution output size
                input_test = torch.randn(1,
                                         self._n_channels,
                                         self._screen_h,
                                         self._screen_w)
                conv_output = self.conv(self.resnet_head(input_test))

                self.conv_output_size = conv_output.view(-1).size(0)

                self.proj1 = nn.Linear(self.conv_output_size, 512)

                self.dropout1 = nn.Dropout(self._dropout)

                self.proj2 = nn.Linear(1024, 128)

                self.dropout2 = nn.Dropout(self._dropout)

                self.proj3 = nn.Linear(128, internal_dim)

            def forward(self, *input):
                pass


        class ImageEncoder(nn.Module):
            def __init__(self, internal_dim, input_dim):
                """
                Image encoder
                Parameters
                ----------
                internal_dim: abstr representation dimension
                input_dim: (in_channels, width, height)
                """
                super(ImageEncoder, self).__init__()
                self._input_dim = input_dim
                self._internal_dim = internal_dim
                # self.c1 = nn.Conv2d(self._input_dim[1], 8, 2, padding=1)
                channels = self._input_dim[1]
                if len(self._input_dim) == 3:
                    channels = self._input_dim[0]
                self.c1 = nn.Conv2d(channels, 8, 3, padding=1)
                self.c2 = nn.Conv2d(8, 16, 3, padding=1, stride=1)
                self.m1 = nn.MaxPool2d(4, padding=1)
                self.c3 = nn.Conv2d(16, 32, 3, stride=1)
                self.m2 = nn.MaxPool2d(3)
                linear_size = 0
                if self._input_dim[-1] == 32:
                    linear_size = 32*2*2
                elif self._input_dim[-1] == 64:
                    linear_size = 64*2*2*2
                else:
                    raise NotImplementedError("Not supported")
                self.linear = nn.Linear(linear_size, self._internal_dim)

                # self.proj = nn.Linear(self._internal_dim, 10)

            def forward(self, x):
                if len(x.shape) > 4:
                    x = x.squeeze(1)
                o1 = torch.tanh(self.c1(x))
                o2 = torch.tanh(self.c2(o1))
                o2 = self.m1(o2)
                o3 = torch.tanh(self.c3(o2))
                o3 = self.m2(o3)
                o4 = o3.view(o3.shape[0], -1)
                out = self.linear(o4)
                # out = F.normalize(out, p=2, dim=-1)

                return out

            def predict(self, x):
                return self(x)

        if self._high_dim_obs:
            # convolutions here
            model = ImageEncoder(self.internal_dim, self._input_dimensions[0])
        else:
            model = VectorEncoder(self.internal_dim,self._input_dimensions)
        
        return model

    def rnn_encoder(self, hidden_size=50):

        class LSTMEncoder(nn.Module):
            def __init__(self, internal_dim, hidden_size=50, num_layers=2):
                super(LSTMEncoder, self).__init__()
                self._internal_dim = internal_dim
                self._hidden_size = hidden_size

                self.lstm = nn.LSTM(self._internal_dim, self._hidden_size, num_layers=num_layers, batch_first=True)

            def forward(self, history):
                if len(history.shape) == 2:
                    history = history.unsqueeze(0)
                out, (h_t, c_t) = self.lstm(history)
                return h_t[-1]

            def predict(self, x):
                return self(x)
        return LSTMEncoder(self.internal_dim, hidden_size=hidden_size)

    def encoder_diff_model(self,encoder_model,s1,s2):
        """ Instantiate a Keras model that provides the difference between two encoded pseudo-states
        
        The model takes the two following inputs:
        s1 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        s2 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder
    
        Returns
        -------
        model with output the difference between the encoding of s1 and the encoding of s2
    
        """


        enc_s1= encoder_model(s1)
        enc_s2= encoder_model(s2)

        
        return enc_s1 - enc_s2

    def transition_model(self):
        """  Instantiate a Keras model for the transition between two encoded pseudo-states.
    
        The model takes as inputs:
        x : internal state concatenated with
            the action considered
        
        Parameters
        -----------
    
        Returns
        -------
        model that outputs the transition of (x,a)
    
        """

        # MLP Transition model
        class Transition(nn.Module):
            def __init__(self, internal_dim, n_actions, dropout_p=0.5, hidden_units=10, scale_network=False):
                super(Transition, self).__init__()

                self._scale_network = scale_network
                if self._scale_network:
                    self.lin1 = nn.Linear(internal_dim + n_actions, 32)
                    self.lin1_2 = nn.Linear(32, hidden_units)
                else:
                    self.lin1 = nn.Linear(internal_dim+n_actions, hidden_units)

                self.lin2 = nn.Linear(hidden_units, hidden_units * 3)
                # self.lin2 = nn.Linear(10, 10)
                self.lin3 = nn.Linear(hidden_units * 3, hidden_units * 3)
                # self.lin3 = nn.Linear(10, 10)
                self.lin4 = nn.Linear(hidden_units * 3, hidden_units)
                # self.lin4 = nn.Linear(10, internal_dim)
                if self._scale_network:
                    self.lin4_5 = nn.Linear(hidden_units, 32)
                    hidden_units = 32
                self.lin5 = nn.Linear(hidden_units, internal_dim)
                self.dropout_p = dropout_p

                self.internal_dim = internal_dim

            def forward(self, x):
                init_state = x[:,:self.internal_dim]
                x = torch.tanh(self.lin1(x))
                x = F.dropout(x, self.dropout_p, training=self.training)

                if self._scale_network:
                    x = torch.tanh(self.lin1_2(x))
                    x = F.dropout(x, self.dropout_p, training=self.training)

                x = torch.tanh(self.lin2(x))
                x = F.dropout(x, self.dropout_p, training=self.training)
                x = torch.tanh(self.lin3(x))

                # x = self.lin4(x)
                x = F.dropout(x, self.dropout_p, training=self.training)
                x = torch.tanh(self.lin4(x))
                x = F.dropout(x, self.dropout_p, training=self.training)
                if self._scale_network:
                    x = torch.tanh(self.lin4_5(x))
                    x = F.dropout(x, self.dropout_p, training=self.training)

                x = self.lin5(x)
                return x + init_state

            def predict(self, x):
                return self.forward(x)

        class MLP(nn.Module):
            """Two-layer fully-connected ELU net with batch norm."""

            def __init__(self, n_in, n_hid, n_out, do_prob=0.):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(n_in, n_hid)
                self.fc2 = nn.Linear(n_hid, n_out)
                self.bn = nn.BatchNorm1d(n_out)
                self.dropout_prob = do_prob

                self.init_weights()

            def init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal(m.weight.data)
                        m.bias.data.fill_(0.1)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def batch_norm(self, inputs):
                x = inputs.view(inputs.size(0) * inputs.size(1), -1)
                x = self.bn(x)
                return x.view(inputs.size(0), inputs.size(1), -1)

            def forward(self, inputs):
                # Input shape: [num_sims, num_things, num_features]
                x = F.elu(self.fc1(inputs))
                x = F.dropout(x, self.dropout_prob, training=self.training)
                x = F.elu(self.fc2(x))
                return x


        # GNN Transition model
        class TransitionGNN(nn.Module):
            def __init__(self, internal_dim, n_actions, n_hid, do_prob=0., factor=True):
                super(TransitionGNN, self).__init__()

                self.internal_dim = internal_dim
                self.n_actions =n_actions

                n_in = 1
                n_out = internal_dim

                self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
                self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
                self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
                # self.mlp4 = MLP(n_hid * 4, n_hid, n_hid, do_prob)
                # self.mlp5 = MLP(n_hid, n_hid, n_hid, do_prob)
                self.fc_out1 = nn.Linear(n_hid*2  * (internal_dim+n_actions), n_hid)
                self.fc_out2 = nn.Linear(n_hid, n_out)
                self.init_weights()

                def encode_onehot(labels):
                    classes = set(labels)
                    classes_dict = {c:  np.identity(len(classes))[i, :] for i, c in
                                    enumerate(classes)}
                    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                             dtype=np.int32)
                    return labels_onehot

                off_diag = np.ones([self.internal_dim+self.n_actions, self.internal_dim+self.n_actions]) - np.eye(self.internal_dim+self.n_actions)
                rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
                rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
                self.rel_rec = torch.FloatTensor(rel_rec)
                self.rel_send = torch.FloatTensor(rel_send)


            def init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal(m.weight.data)
                        m.bias.data.fill_(0.1)

            def edge2node(self, x):
                # NOTE: Assumes that we have the same graph across all samples.
                incoming = torch.matmul(self.rel_rec.t(), x)
                return incoming / incoming.size(1)

            def node2edge(self, x):
                # NOTE: Assumes that we have the same graph across all samples.
                receivers = torch.matmul(self.rel_rec, x)
                senders = torch.matmul(self.rel_send, x)
                edges = torch.cat([receivers, senders], dim=2)
                return edges

            def forward(self, inputs):
                # import pdb;pdb.set_trace()
                
                init_state = inputs[:,:self.internal_dim]
                x = inputs.view(inputs.size(0), inputs.size(1), 1)
                x = self.mlp1(x)  # 2-layer ELU net per node
                x_skip = x

                x = self.node2edge(x)
                x = self.mlp2(x)
                
                x = self.edge2node(x)
                x = self.mlp3(x)

                x = torch.cat((x, x_skip), dim=2)



                x = x.view(x.size(0), -1)
                x= F.elu(self.fc_out1(x))
                x= self.fc_out2(x)
                return x + init_state

            def predict(self, x):
                return self.forward(x)


        model = Transition(self.internal_dim,self._n_actions, dropout_p=self._transition_dropout_p,
                           hidden_units=self._transition_hidden_units, scale_network=self._scale_transition)
        # model = TransitionGNN(self.internal_dim, self._n_actions, 32)



        return model

    def diff_Tx_x_(self,s1,s2,action,not_terminal,encoder_model,transition_model,plan_depth=0):
        """ For plan_depth=0, instantiate a Keras model that provides the difference between T(E(s1),a) and E(s2).
        Note that it gives 0 if the transition leading to s2 is terminal (we don't need to fit the transition if 
        it is terminal).
        
        For plan_depth=0, the model takes the four following inputs:
        s1 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        s2 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        1 : list of ints with length (plan_depth+1)
            the action(s) considered at s1
        terminal : boolean
            Whether the transition leading to s2 is terminal
        
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        transition_model: instantiation of a Keras model for the transition (T)
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions between s1 and s2 
        (input a is then a list of actions)
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """


        enc_s1 = encoder_model(s1)
        enc_s2 = encoder_model(s2)

        Tx = transition_model(torch.cat((enc_s1, action), dim=-1))

        return (Tx - enc_s2)*(not_terminal)

    def force_features(self,s1,s2,action,encoder_model,transition_model,plan_depth=0):
        """ Instantiate a PyTorch model that provides the vector of the transition at E(s1). It is calculated as the different between E(s1) and E(T(s1)).
        Used to force the directions of the transitions.
        
        The model takes the four following inputs:
        s1 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length (plan_depth+1)
            the action(s) considered at s1
        
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        transition_model: instantiation of a Keras model for the transition (T)
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions between s1 and s2 
        (input a is then a list of actions)
            
        Returns
        -------
        model with output E(s1)-T(E(s1))
    
        """


        enc_s1 = encoder_model(s1)
        enc_s2 = encoder_model(s2)

        Tx = transition_model(torch.cat((enc_s1,action),-1))


        return (Tx - enc_s2)


    def float_model(self):
        """ Instantiate a Keras model for fitting a float from x.
                
        The model takes the following inputs:
        x : internal state
        a : int
            the action considered at x
        
        Parameters
        -----------
            
        Returns
        -------
        model that outputs a float
    
        """
        

        class FloatModel(nn.Module):
            def __init__(self,internal_dim,n_actions):
                super(FloatModel, self).__init__()
                self.lin1 = nn.Linear(internal_dim+n_actions, 10)
                self.lin2 = nn.Linear(10, 50)
                self.lin3 = nn.Linear(50, 20)
                self.lin4 = nn.Linear(20, 1)

            def forward(self, x):

                x = torch.tanh(self.lin1(x))
                x = torch.tanh(self.lin2(x))
                x = torch.tanh(self.lin3(x))
                x = self.lin4(x)
                return x
            def predict(self, x):
                return self.forward(x)

        model = FloatModel(self.internal_dim,self._n_actions)



        return model

    def full_float_model(self,x,action,encoder_model,float_model,plan_depth=0,transition_model=None):
        """ Instantiate a Keras model for fitting a float from s.
                
        The model takes the four following inputs:
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length (plan_depth+1)
            the action(s) considered at s
                
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        float_model: instantiation of a Keras model for fitting a float from x
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s 
        (input a is then a list of actions)
        transition_model: instantiation of a Keras model for the transition (T)
            
        Returns
        -------
        model with output the reward r
        """
        

        enc_x = encoder_model(x)
        reward_pred = float_model(torch.cat((enc_x,action),-1))
        return reward_pred

    def RNN_Q_model(self, encoder_model, use_representation=True):
        """
        RNN Q model that encodes history and uses that as an input to a Q function approximator.
        NOTE: must use set_history before calling forward.
        :param use_representation: use low dimensional input. Currently only available option.
        :return: model with output Q value
        """
        assert use_representation == True
        class LSTMEncoder(nn.Module):
            def __init__(self, internal_dim, hidden_size=50, num_layers=2):
                super(LSTMEncoder, self).__init__()
                self._internal_dim = internal_dim
                self._hidden_size = hidden_size

                self.lstm = nn.LSTM(self._internal_dim, self._hidden_size, num_layers=num_layers, batch_first=True)

            def forward(self, history):
                if len(history.shape) == 2:
                    history = history.unsqueeze(0)
                out, (h_t, c_t) = self.lstm(history)
                return h_t[-1]

            def predict(self, x):
                return self(x)

        class LowDimQFunction(nn.Module):
            def __init__(self,input_dim,n_actions):
                super(LowDimQFunction, self).__init__()
                self.lin1 = nn.Linear(input_dim, 100)
                self.lin2 = nn.Linear(100, 50)
                self.lin3 = nn.Linear(50, 20)
                self.lin4 = nn.Linear(20, n_actions)

            def forward(self, x):
                x = torch.relu(self.lin1(x))
                x = torch.relu(self.lin2(x))
                x = torch.relu(self.lin3(x))
                x = self.lin4(x)
                return x
            def predict(self, x):
                return self.forward(x)

        class RNN_Q_Func(nn.Module):
            def __init__(self, internal_dim, n_actions,
                         hidden_size=50, initial_history=None):
                super(RNN_Q_Func, self).__init__()
                self.lstm_encoder = LSTMEncoder(internal_dim, hidden_size=hidden_size)
                self.Q = LowDimQFunction(internal_dim + hidden_size, n_actions)
                self.history = initial_history

            def set_history(self, history):
                self.history = history

            def forward(self, x):
                encoder_model.eval()
                history_inp = torch.from_numpy(self.history).float().to(device)
                with torch.no_grad():
                    out = encoder_model(history_inp)

                hs = self.lstm_encoder(out)
                if hs.shape[0] != x.shape[0]:
                    hs = hs.repeat(x.shape[0], 1)
                inp = torch.cat((hs, x), dim=-1)
                return self.Q(inp)

            def predict(self, x):
                return self(x)

        return RNN_Q_Func(self.internal_dim, self.n_actions)

    def Q_model(self, use_representation=True):
        """ Instantiate a model for the Q-network from x.

        The model takes the following inputs:
        x : internal state

        Parameters
        -----------
            
        Returns
        -------
        model that outputs the Q-values for each action
        """

        class VectorQFunction(nn.Module):
            def __init__(self, input_dim, n_actions, headless=False):
                super(VectorQFunction, self).__init__()
                self.input_dim_flat = np.prod(input_dim)
                self.lin1 = nn.Linear(self.input_dim_flat, 500)
                self.lin2 = nn.Linear(500, 200)
                self.lin3 = nn.Linear(200, 50)
                self.lin4 = nn.Linear(50, 10)
                self.output_dim = 10
                self.lin5 = None
                self._headless = headless
                if not self._headless:
                    self.lin5 = nn.Linear(self.output_dim, n_actions)

            def forward(self, x):
                x = x.view(-1, self.input_dim_flat)
                x = torch.tanh(self.lin1(x))
                x = torch.tanh(self.lin2(x))
                x = torch.tanh(self.lin3(x))
                x = torch.tanh(self.lin4(x))
                if not self._headless:
                    x = self.lin5(x)
                return x

        class BootstrappedQFunction(nn.Module):
            def __init__(self, input_dim, n_actions, n_heads=10,
                         append_hidden_dims=[],
                         append_hidden_init_func=inits.basic_init,
                         last_init_func=inits.uniform_init, image_input=False):
                """
                Taken from https://github.com/RchalYang/torchrl
                :param input_dim:
                :param n_actions:
                :param n_heads:
                :param append_hidden_dims:
                :param append_hidden_init_func:
                :param last_init_func:
                :param image_input:
                """
                super(BootstrappedQFunction, self).__init__()
                self.n_heads = n_heads
                self._n_actions = n_actions
                self._input_dimensions = input_dim
                self._image_input = image_input
                self.base = None
                if use_representation:
                    self.base = LowDimQFunction(self._input_dimensions, self._n_actions, headless=True)
                elif self._image_input:
                    self.base = ImageQFunction(self._input_dimensions, self._n_actions, headless=True)
                else:
                    self.base = VectorQFunction(self._input_dimensions, self._n_actions, headless=True)
                self.last_append_layers = []
                self.last_layers = []
                assert self.n_heads > 0
                for i in range(self.n_heads):
                    append_input_dim = self.base.output_dim
                    append_layers = []
                    for j, next_dim in enumerate(append_hidden_dims):
                        layer = nn.Linear(append_input_dim, next_dim)
                        append_hidden_init_func(layer)
                        append_layers.append(layer)
                        self.__setattr__("head_{}_append_fc{}".format(i, j), layer)
                        append_input_dim = next_dim

                    last = nn.Linear(append_input_dim, self._n_actions)
                    last_init_func(last)
                    self.last_layers.append(last)
                    self.__setattr__("head_{}_last".format(i), last)

                    self.last_append_layers.append(append_layers)

            def forward(self, x, head_idxs):
                output = []
                features = self.base(x)
                for idx in head_idxs:
                    out = features
                    for append_fc in self.last_append_layers[ idx ]:
                        out = append_fc(out)
                        out = self.activation_func(out)
                    out = self.last_layers[idx](out)
                    output.append(out)
                return output

        class ImageQFunction(nn.Module):
            def __init__(self, input_dim, n_actions, headless=False):
                super(ImageQFunction, self).__init__()

                self._input_dim = input_dim
                self._n_actions = n_actions
                channels = input_dim[0][0]
                if len(input_dim[0]) == 4:
                    channels = input_dim[0][1]
                self.c1 = nn.Conv2d(channels, 8, 3, padding=1)

                self.c2 = nn.Conv2d(8, 16, 3, padding=1, stride=1)
                self.m1 = nn.MaxPool2d(4, padding=1)
                self.c3 = nn.Conv2d(16, 32, 3, stride=1)
                self.m2 = nn.MaxPool2d(3)
                self.output_dim = 32 * 2 * 2
                self.linear = None
                self._headless = headless
                if not self._headless:
                    self.linear = nn.Linear(self.output_dim, self._n_actions)

            def forward(self, x):
                if len(x.shape) < 4:
                    x = x.unsqueeze(1)

                o1 = torch.relu(self.c1(x))
                o2 = torch.relu(self.c2(o1))
                o2 = self.m1(o2)
                o3 = torch.relu(self.c3(o2))
                o3 = self.m2(o3)
                o = o3.view(o3.shape[0], -1)
                if not self._headless:
                    o = self.linear(o)

                return o

            def predict(self, x):
                return self(x)


        class LowDimQFunction(nn.Module):
            def __init__(self,internal_dim,n_actions, headless=False):
                super(LowDimQFunction, self).__init__()
                self.lin1 = nn.Linear(internal_dim, 20)
                self.lin2 = nn.Linear(20, 50)
                self.lin3 = nn.Linear(50, 20)
                self.output_dim = 20
                self.lin4 = None
                self._headless = headless
                if not self._headless:
                    self.lin4 = nn.Linear(self.output_dim, n_actions)

            def forward(self, x):
                x = torch.relu(self.lin1(x))
                x = torch.relu(self.lin2(x))
                x = torch.relu(self.lin3(x))
                if not self._headless:
                    x = self.lin4(x)
                return x
            def predict(self, x):
                return self.forward(x)

        headless = self._bootstrap_q_func

        if self._bootstrap_q_func:
            input_dim = self._input_dimensions if not use_representation else self.internal_dim
            model = BootstrappedQFunction(input_dim, self._n_actions, image_input=self._high_dim_obs)
        else:
            if use_representation:
                model = LowDimQFunction(self.internal_dim,self._n_actions, headless=headless)
            else:
                if not self._high_dim_obs:
                    model = VectorQFunction(self._input_dimensions, self._n_actions, headless=headless)
                else:
                    model = ImageQFunction(self._input_dimensions, self._n_actions)
                # model = LowDimQFunctionCheck(2, self._n_actions)
        return model


    def full_Q_model(self, x, encoder_model, Q_model, static_encoder=False):
        """ Instantiate a Pytorch model for the Q-network from s.

        The model takes the following inputs:
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length plan_depth; if plan_depth=0, there isn't any input for a.
            the action(s) considered at s
    
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        Q_model: instantiation of a Keras model for the Q-network from x.
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s 
        (input a is then a list of actions)
        transition_model: instantiation of a Keras model for the transition (T)
        R_model: instantiation of a Keras model for the reward
        discount_model: instantiation of a Keras model for the discount
            
        Returns
        -------
        model with output the Q-values
        """

        if static_encoder:
            with torch.no_grad():
                out = encoder_model(x)
        else:
            out = encoder_model(x)
        # FIXME
        Q_estim = Q_model(out)

        return Q_estim

    @property
    def n_actions(self):
        return self._n_actions

if __name__ == '__main__':
    pass
    