import torch
import torch.nn as nn


class LSTM_NodeLinWeightsDistribution(nn.Module):
    """
    Linear structural equation autoregressive weights distribution for a node given its parents using an LSTM
    """
    def __init__(self, n_weights, hidden_dim=48, n_layers=3):
        """
        Initialise distribution
        :param n_weights: number of weights to be modelled by the distribution. It needs to be equivalent to the number
                          of nodes in the graph.
        :param hidden_dim: LSTM input, hidden and cell states dimension
        :param n_layers: number of layers in the LSTM
        """
        super().__init__()
        self.n_weights = n_weights
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=n_layers, batch_first=True)
        # Project the output of the LSTM into a mean
        self.proj_mean = nn.Linear(self.hidden_dim, 1)
        # Embed the input of the LSTM into a vector
        self.embed_input = nn.Linear(1, self.hidden_dim)
        # Embed an adjacency entry into a vector
        self.embed_adj = nn.Linear(self.n_weights, self.hidden_dim)
        # Initialise the initial hidden state. It will be updated when using back-propagation
        self.h0 = nn.Parameter(1e-3 * torch.randn(1, self.n_layers, self.hidden_dim))
        # Initialise the initial cell state. It will be updated when using back-propagation
        self.c0 = nn.Parameter(1e-3 * torch.randn(1, self.n_layers, self.hidden_dim))
        # Initialise variable for the initial input of the LSTM. It will be updated when using back-propagation
        self._init_input_param = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, adj, reparametrized=True, return_norm_params=False, start_state=None, init_input=None):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent
        :param adj: batch of adjacency entries
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :param return_norm_params: if True, the Normal distribution parameters are returned as well
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: weight vector samples or weight vectors samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=reparametrized, start_state=start_state,
                                            init_input=init_input)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _compute_norm_params(self, adj, inputs, state):
        """
        Compute batch of means, stds and states by feeding a batch of adjacency entries, inputs and initial states to
        the LSTM
        :param adj: batch of adjacency entries
        :param inputs: batch of inputs to the LSTM
        :param state: batch of initial states of the LSTM
        :return: batch of means, stds, output states
        """
        # adj size = (B,T) where B=batch size, T=self.n_weights
        # input size = (B,L,1) where B=batch size, L=sequence length,
        # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
        inputs = self.embed_input(inputs)
        # inputs size = (B,L,H) where B=batch size, L=sequence length, H=self.hidden_dim
        adj = self.embed_adj(adj)
        # adj size = (B,H) where B=batch size, H=self.hidden_dim
        adj = adj.unsqueeze(1)
        # adj size = (B,1,H) where B=batch size, H=self.hidden_dim
        inputs = inputs * adj
        # inputs size = (B,L,H)
        out, state = self.rnn(inputs, self._t(state))
        # out size = (B,L,H) where B=batch size, L=sequence length, H=self.hidden_dim
        # state = tuple where both elements have size (Q,B,H) where Q=self.n_layers, B=batch size, H=self.hidden_dim
        state = self._t(state)
        # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
        means = self.proj_mean(out)
        # means size = (B,L,1) where B=batch size, L=sequence length
        stds = torch.ones_like(means)
        # stds size = (B,L,1) where B=batch size, L=sequence length
        return means, stds, state

    def sample(self, adj, return_norm_params=False, start_state=None, init_input=None):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent, using the Normal distribution.
        :param adj: batch of adjacency entries
        :param return_norm_params: if True, the Normal distribution parameters are returned as well
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: weight vector samples or weight vectors samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, start_state=start_state, init_input=init_input)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def rsample(self, adj, return_norm_params=False, start_state=None, init_input=None):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent, using the Normal distribution reparameterization trick.
        :param adj: batch of adjacency entries
        :param return_norm_params: if True, the Normal distribution parameters are returned as well
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: weight vector samples or weight vectors samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=True, start_state=start_state,
                                            init_input=init_input)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _sample(self, adj, reparametrized=False, start_state=None, init_input=None):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: weight vectors samples, means and stds
        """
        assert adj.shape[1] == self.n_weights
        # adj size = (B,T) where B=batch size, T=self.n_weights
        if start_state is None:
            state = self._get_state(adj.shape[0])  # hidden / cell state at t=0
        else:
            state = start_state
        if init_input is None:
            input = self._get_init_input(adj.shape[0])  # input at t=0
        else:
            input = init_input

        # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
        # input size = (B,1,1) where B=batch size

        sampled_tokens = []
        state_array_1 = []
        state_array_2 = []
        means_array = []
        stds_array = []

        for t in range(self.n_weights):
            means, stds, state = self._compute_norm_params(adj, input, state)
            # means size = (B,1,1) where B=batch size
            # stds size = (B,1,1) where B=batch size
            # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
            if reparametrized:
                _sample = torch.distributions.normal.Normal(means, stds).rsample().unsqueeze(2)
                # _sample size = (B,1,1) where B=batch size
            else:
                _sample = torch.distributions.normal.Normal(means, stds).sample().unsqueeze(2)
                # _sample size = (B,1,1) where B=batch size
            adj_col = adj[:, t].unsqueeze(2)
            # adj_col size = (B,1,1) where B=batch size
            _sample = _sample * adj_col
            # _sample size = (B,1,1) where B=batch size
            means = means * adj_col
            # means size = (B,1,1) where B=batch size
            stds = stds * adj_col
            # stds size = (B,1,1) where B=batch size
            input = _sample
            sampled_tokens.append(_sample)
            state_array_1.append(state[0])
            state_array_2.append(state[1])
            means_array.append(means)
            stds_array.append(stds)

        samples = torch.cat(sampled_tokens, dim=1)
        # samples size = (B,T,1) where B=batch size, T=self.n_weights
        samples = samples.squeeze(2)
        # samples size = (B,T) where B=batch size, T=self.n_weights
        states = [torch.stack(state_array_1, dim=1), torch.stack(state_array_2, dim=1)]
        # states = tuple where both elements have size (B,T,Q,H) where B=batch size, T=self.n_dim_out, Q=self.n_layers,
        # H=self.hidden_dim
        means = torch.cat(means_array, dim=1)
        # means size = (B,T,1) where B=batch size, T=self.n_weights
        means = means.squeeze(2)
        # means size = (B,T) where B=batch size, T=self.n_weights
        stds = torch.cat(stds_array, dim=1)
        # stds size = (B,T,1) where B=batch size, T=self.n_weights
        stds = stds.squeeze(2)
        # stds size = (B,T) where B=batch size, T=self.n_weights
        return samples, means, stds

    def _get_state(self, batch_size=1):
        """
        Get a batch of initial states. The initial state is just repeated n times where n is the batch size value.
        :param batch_size: batch size
        :return: batch of initial states
        """
        return self.h0.repeat(batch_size, 1, 1), self.c0.repeat(batch_size, 1, 1)

    def _get_init_input(self, batch_size):
        """
        Get a batch of the initial input. The initial input is just repeated n times where n is the batch size value.
        :param batch_size: batch size
        :return: batch of the initial input
        """
        return self._init_input_param.expand(batch_size, 1, 1)

    @staticmethod
    def _t(a):
        return [t.transpose(0, 1).contiguous() for t in a]