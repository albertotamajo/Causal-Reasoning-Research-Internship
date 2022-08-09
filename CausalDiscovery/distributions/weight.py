import torch
import torch.nn as nn

import utils


class LSTM_NodeLinWeightsDistribution(nn.Module):
    # TODO: Recheck implementation and implement log, mean and mode functions
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
                _sample = torch.distributions.normal.Normal(means, stds).rsample()
                # _sample size = (B,1,1) where B=batch size
            else:
                _sample = torch.distributions.normal.Normal(means, stds).sample()
                # _sample size = (B,1,1) where B=batch size
            adj_col = (adj[:, t]).unsqueeze(1).unsqueeze(2)
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


class MLP_NodeLinWeightsDistribution(nn.Module):
    """
    Linear structural equation weights distribution for a node given its parents using a single MLP
    """

    def __init__(self, n_weights):
        """
        Initialise distribution
        :param n_weights: number of weights to be modelled by the distribution. It needs to be equivalent to the number
                          of nodes in the graph
        """
        super().__init__()
        self.n_weights = n_weights
        self.proj_mean = nn.Sequential(nn.Linear(n_weights, n_weights + 10), nn.ReLU(),
                                       nn.Linear(n_weights + 10, n_weights))

    def forward(self, adj, reparametrized=True, return_norm_params=False):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight vector samples or weight vector samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=reparametrized)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _compute_norm_params(self, adj):
        """
        Compute batch of Normal distribution parameters by feeding a batch of adjacency entries, where these entries
        denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :return: batch of means and stds
        """
        # adj size = (B,Q) where B=batch size, Q=n_weights
        means = self.proj_mean(adj)
        # means size = (B,Q) where B=batch size, Q=n_weights
        stds = torch.ones_like(means)
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        return means, stds

    def sample(self, adj, return_norm_params=False):
        """
        Sample a batch of weight vectors from the distribution given a batch of adjacency entries using the Normal
        distribution.
        :param adj: batch of adjacency entries
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight vector samples or weight vector samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=False)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def rsample(self, adj, return_norm_params=False):
        """
        Sample a batch of weight vectors from the distribution given a batch of adjacency entries using the Normal
        distribution reparameterization trick.
        :param adj: batch of adjacency entries
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight vector samples or weight vector samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=True)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _sample(self, adj, reparametrized=False):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :return: weight vectors samples, means and stds
        """
        assert adj.shape[1] == self.n_weights
        # adj size = (B,Q) where B=batch size, Q=self.n_weights
        means, stds = self._compute_norm_params(adj)
        # means size = (B,Q) where B=batch size, Q=self.n_weights
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        if reparametrized:
            _sample = torch.distributions.normal.Normal(means, stds).rsample()
            # _sample size = (B,Q) where B=batch size, Q=self.n_weights
        else:
            _sample = torch.distributions.normal.Normal(means, stds).sample()
            # _sample size = (B,Q) where B=batch size, Q=self.n_weights
        _sample = _sample * adj
        # _sample size = (B,Q) where B=batch size, Q=self.n_weights
        means = means * adj
        stds = stds * adj
        return _sample, means, stds

    def log_prob(self, weight_vecs, adj, return_probs=False):
        """
        Compute the log probability of the given batch of weight vectors given a batch of adjacency entries, where
        these entries denote whether a given node is a parent.
        :param weight_vecs: batch of weight vectors
        :param adj: batch of adjacency entries
        :param return_probs: if True the Normal distribution parameters for each entry of the weight vectors is
                             returned as well
        :return: log probability for the batch of weight vectors or log probability for the batch of weight vectors
                 matrices and the Normal distribution parameters for each entry of the weight vectors
        """
        assert adj.shape[1] == self.n_weights and weight_vecs.shape[1] == self.n_weights
        # weight_vecs size = (B,Q) where B=batch size, Q=self.n_weights
        # adj size = (B,Q) where B=batch size, Q=self.n_weights
        means, stds = self._compute_norm_params(adj)
        # means size = (B,Q) where B=batch size, Q=self.n_weights
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        log_probs = torch.distributions.normal.Normal(means, stds).log_prob(weight_vecs) * adj
        # log_probs size = (B,Q) where B=batch size, Q=self.n_weights
        log_probs = log_probs.sum(1, keepdim=True)
        # log_probs size = (B,1)
        if return_probs:
            return log_probs, means * adj, stds * adj
        return log_probs

    def mode(self, adj, return_stds=False):
        """
        Compute a batch of weight vector modes given a batch of adjacency entries, where
        these entries denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param return_stds: if True, the stds parameters are returned as well
        :return: batch of weight vector modes or batch of weight vector modes, stds
        """
        assert adj.shape[1] == self.n_weights
        # adj size = (B,Q) where B=batch size, Q=self.n_weights
        means, stds = self._compute_norm_params(adj)
        means = means * adj
        stds = stds * adj
        # means size = (B,Q) where B=batch size, Q=self.n_weights
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        if return_stds:
            return means, stds
        else:
            return means

    def mean(self, adj, return_stds=False):
        """
        Compute a batch of weight vector means given a batch of adjacency entries, where
        these entries denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param return_stds: if True, the stds parameters are returned as well
        :return: batch of weight vector means or batch of weight vector means, stds
        """
        return self.mode(adj, return_stds)


class MLPs_NodeLinWeightsDistribution(nn.Module):
    """
    Linear structural equation weights distribution for a node given its parents. Each weight distribution is modelled
    using an MLP.
    """
    def __init__(self, n_weights):
        """
        Initialise distribution
        :param n_weights: number of weights to be modelled by the distribution. It needs to be equivalent to the number
                          of nodes in the graph
        """
        super().__init__()
        self.n_weights = n_weights
        self.proj_mean = nn.ModuleList([nn.Linear(n_weights, 1) for _ in range(self.n_weights)])

    def forward(self, adj, reparametrized=True, return_norm_params=False):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight vector samples or weight vector samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=reparametrized)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _compute_norm_params(self, adj):
        """
        Compute batch of Normal distribution parameters by feeding a batch of adjacency entries, where these entries
        denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :return: batch of means and stds
        """
        # adj size = (B,Q) where B=batch size, Q=n_weights
        means = []
        for i in range(self.n_weights):
            node_means = self.proj_mean[i](adj)
            # node_means size = (B,1) where B=batch size
            means.append(node_means)
        means = torch.cat(means, dim=1)
        # means size = (B,Q) where B=batch size, Q=n_weights
        stds = torch.ones_like(means).fill_(0.0001)
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        return means, stds

    def sample(self, adj, return_norm_params=False):
        """
        Sample a batch of weight vectors from the distribution given a batch of adjacency entries using the Normal
        distribution.
        :param adj: batch of adjacency entries
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight vector samples or weight vector samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=False)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def rsample(self, adj, return_norm_params=False):
        """
        Sample a batch of weight vectors from the distribution given a batch of adjacency entries using the Normal
        distribution reparameterization trick.
        :param adj: batch of adjacency entries
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight vector samples or weight vector samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=True)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _sample(self, adj, reparametrized=False):
        """
        Sample a batch of weight vectors given a batch of adjacency entries, where these entries
        denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :return: weight vectors samples, means and stds
        """
        assert adj.shape[1] == self.n_weights
        # adj size = (B,Q) where B=batch size, Q=self.n_weights
        means, stds = self._compute_norm_params(adj)
        # means size = (B,Q) where B=batch size, Q=self.n_weights
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        if reparametrized:
            _sample = torch.distributions.normal.Normal(means, stds).rsample()
            # _sample size = (B,Q) where B=batch size, Q=self.n_weights
        else:
            _sample = torch.distributions.normal.Normal(means, stds).sample()
            # _sample size = (B,Q) where B=batch size, Q=self.n_weights
        _sample = _sample * adj
        # _sample size = (B,Q) where B=batch size, Q=self.n_weights
        means = means * adj
        stds = stds * adj
        return _sample, means, stds

    def log_prob(self, weight_vecs, adj, return_probs=False):
        """
        Compute the log probability of the given batch of weight vectors given a batch of adjacency entries, where
        these entries denote whether a given node is a parent.
        :param weight_vecs: batch of weight vectors
        :param adj: batch of adjacency entries
        :param return_probs: if True the Normal distribution parameters for each entry of the weight vectors is
                             returned as well
        :return: log probability for the batch of weight vectors or log probability for the batch of weight vectors
                 matrices and the Normal distribution parameters for each entry of the weight vectors
        """
        assert adj.shape[1] == self.n_weights and weight_vecs.shape[1] == self.n_weights
        # weight_vecs size = (B,Q) where B=batch size, Q=self.n_weights
        # adj size = (B,Q) where B=batch size, Q=self.n_weights
        means, stds = self._compute_norm_params(adj)
        # means size = (B,Q) where B=batch size, Q=self.n_weights
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        log_probs = torch.distributions.normal.Normal(means, stds).log_prob(weight_vecs) * adj
        # log_probs size = (B,Q) where B=batch size, Q=self.n_weights
        log_probs = log_probs.sum(1, keepdim=True)
        # log_probs size = (B,1)
        if return_probs:
            return log_probs, means * adj, stds * adj
        return log_probs

    def mode(self, adj, return_stds=False):
        """
        Compute a batch of weight vector modes given a batch of adjacency entries, where
        these entries denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param return_stds: if True, the stds parameters are returned as well
        :return: batch of weight vector modes or batch of weight vector modes, stds
        """
        assert adj.shape[1] == self.n_weights
        # adj size = (B,Q) where B=batch size, Q=self.n_weights
        means, stds = self._compute_norm_params(adj)
        means = means * adj
        stds = stds * adj
        # means size = (B,Q) where B=batch size, Q=self.n_weights
        # stds size = (B,Q) where B=batch size, Q=self.n_weights
        if return_stds:
            return means, stds
        else:
            return means

    def mean(self, adj, return_stds=False):
        """
        Compute a batch of weight vector means given a batch of adjacency entries, where
        these entries denote whether a given node is a parent.
        :param adj: batch of adjacency entries
        :param return_stds: if True, the stds parameters are returned as well
        :return: batch of weight vector means or batch of weight vector means, stds
        """
        return self.mode(adj, return_stds)


class LinWeightsDistribution(nn.Module):
    """
    Conditional linear structural equation weights distribution given an adjacency matrix.
    """
    def __init__(self, n_nodes, node_dist):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        :param node_dist: reference to a class that models the weight distribution for a given node.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.lin_weights_distribution_nodes = nn.ModuleList([node_dist(self.n_nodes)
                                                            for _ in range(self.n_nodes)])

    def forward(self, adj, reparametrized=True, return_norm_params=False):
        """
        Sample a batch of weights given a batch of adjacency matrices
        :param adj: batch of adjacency matrices
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight samples or weight samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=reparametrized)
        if return_norm_params:
            return samples, means, stds
        return samples

    def sample(self, adj, return_norm_params=False):
        """
        Sample a batch of weights from the distribution given a batch of adjacency matrices using the Normal
        distribution
        :param adj: batch of adjacency matrices
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight samples or weight samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=False)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def rsample(self, adj, return_norm_params=False):
        """
        Sample a batch of weights from the distribution given a batch of adjacency matrices using the Normal
        distribution reparameterization trick.
        :param adj: batch of adjacency matrices
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight samples or weight samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=True)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _sample(self, adj, reparametrized=False):
        """
        Sample a batch of weights given a batch of adjacency matrices
        :param adj: batch of adjacency entries
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :return: weight samples, means and stds
        """
        # adj size = (B,T,T) where B=batch size, T=self.n_nodes
        assert adj.shape[1] == self.n_nodes and adj.shape[2] == self.n_nodes
        sample_list = []
        means_list = []
        stds_list = []
        for i in range(self.n_nodes):
            _sample, means, stds = self.lin_weights_distribution_nodes[i](adj[:, i, :], reparametrized=reparametrized,
                                                                          return_norm_params=True)
            # _sample size = (B,T) where B=batch size, T=self.n_nodes
            # means size = (B,T) where B=batch size, T=self.n_nodes
            # stds size = (B,T) where B=batch size, T=self.n_nodes
            _sample = _sample.unsqueeze(1)
            # _sample size = (B,1,T) where B=batch size, T=self.n_nodes
            means = means.unsqueeze(1)
            # means size = (B,1,T) where B=batch size, T=self.n_nodes
            stds = stds.unsqueeze(1)
            # stds size = (B,1,T) where B=batch size, T=self.n_nodes
            sample_list.append(_sample)
            means_list.append(means)
            stds_list.append(stds)
        samples = torch.cat(sample_list, dim=1)
        # samples size = (B,T,T) where B=batch size, T=self.n_nodes
        means = torch.cat(means_list, dim=1)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        stds = torch.cat(stds_list, dim=1)
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        return samples, means, stds

    def log_prob(self, weights, adj, return_probs=False):
        """
        Compute the log probability of the given batch of weights given a batch of adjacency matrices
        :param weights: batch of weights
        :param adj: batch of adjacency matrices
        :param return_probs: if True the Normal distribution parameters for each entry of the weights is
                             returned as well
        :return: log probability for the batch of weights or log probability for the batch of weights,
                 ,the Normal distribution parameters for each entry of the weights
        """
        # weights size = (B,T,T) where B=batch size, T=self.n_nodes
        # adj size = (B,T,T) where B=batch size, T=self.n_nodes
        assert [adj.shape[1], adj.shape[2]] == [self.n_nodes] * 2 \
               and [weights.shape[1], weights.shape[2]] == [self.n_nodes] * 2
        log_probs_list = []
        means_list = []
        stds_list = []
        for i in range(self.n_nodes):
            log_probs, means, stds = self.lin_weights_distribution_nodes[i].log_prob(weights[:, i, :], adj[:, i, :],
                                                                                     return_probs=True)
            # log_probs size = (B,1) where B=batch size
            # means size = (B,T) where B=batch size, T=self.n_nodes
            # stds size = (B,T) where B=batch size, T=self.n_nodes
            log_probs = log_probs.unsqueeze(1)
            # log_probs size = (B,1,1) where B=batch size
            means = means.unsqueeze(1)
            # means size = (B,1,T) where B=batch size
            stds = stds.unsqueeze(1)
            # stds size = (B,1,T) where B=batch size
            log_probs_list.append(log_probs)
            means_list.append(means)
            stds_list.append(stds)
        log_probs = torch.cat(log_probs_list, dim=1).squeeze(2)
        # log_probs size = (B,T) where B=batch size, T=self.n_nodes
        log_probs = log_probs.sum(1, keepdim=True)
        # log_probs size = (B,1) where B=batch size
        means = torch.cat(means_list, dim=1)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        stds = torch.cat(stds_list, dim=1)
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        if return_probs:
            return log_probs, means, stds
        return log_probs

    def mode(self, adj, return_stds=False):
        """
        Compute a batch of weights modes given a batch of adjacency matrices
        :param adj: batch of adjacency matrices
        :param return_stds: if True, the stds parameters are returned as well
        :return: batch of weights modes or batch of weight modes, stds
        """
        # adj size = (B,T,T) where B=batch size, T=self.n_nodes
        assert [adj.shape[1], adj.shape[2]] == [self.n_nodes] * 2
        _, means, stds = self._sample(adj)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        if return_stds:
            return means, stds
        else:
            return means

    def mean(self, adj, return_stds=False):
        """
        Compute a batch of weights means given a batch of adjacency matrices
        :param adj: batch of adjacency matrices
        :param return_stds: if True, the stds parameters are returned as well
        :return: batch of weights means or batch of weights means, stds
        """
        return self.mode(adj, return_stds)


class LinWeightsMatrixDistribution(nn.Module):
    # TODO: RECHECK IMPLEMENTATION, IMPLEMENT LOG PROB, MEANS AND MODE FUNCTIONS
    """
    Conditional linear structural equation weights distribution given an adjacency matrix. The means of the weights
    are simply stored as learnable parameters in a matrix.
    """
    def __init__(self, n_nodes, init_params=None):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        :param init_params: if None, the parameters are initialised randomly. Otherwise, a tensor
                            of shape (n_nodes * (n_nodes - 1), 1) should be provided to initialise the parameters.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim_out = n_nodes * (n_nodes - 1)
        if init_params is None:
            self.means = torch.nn.Parameter(torch.randn(self.n_dim_out, 1))
            # params size = (Q,1) where Q=self.n_dim_out
        else:
            assert init_params.shape == (self.n_dim_out, 1)
            self.means = torch.nn.Parameter(init_params)
            # params size = (Q,1) where Q=self.n_dim_out

    def forward(self, adj, reparametrized=True, return_norm_params=False):
        """
        Sample a batch of weights given a batch of adjacency matrices
        :param adj: batch of adjacency matrices
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight samples or weight samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=reparametrized)
        if return_norm_params:
            return samples, means, stds
        return samples

    def sample(self, adj, return_norm_params=False):
        """
        Sample a batch of weights from the distribution given a batch of adjacency matrices using the Normal
        distribution
        :param adj: batch of adjacency matrices
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight samples or weight samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=False)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def rsample(self, adj, return_norm_params=False):
        """
        Sample a batch of weights from the distribution given a batch of adjacency matrices using the Normal
        distribution reparameterization trick.
        :param adj: batch of adjacency matrices
        :param return_norm_params: if True, the Normal distribution parameters are returned
        :return: weight samples or weight samples, Normal distribution parameters
        """
        samples, means, stds = self._sample(adj, reparametrized=True)
        if return_norm_params:
            return samples, means, stds
        else:
            return samples

    def _sample(self, adj, reparametrized=False):
        """
        Sample a batch of weights given a batch of adjacency matrices
        :param adj: batch of adjacency matrices
        :param reparametrized: if False, every element of the weight vectors is sampled using a Normal distribution.
                               Sampling from the Normal distribution is not differentiable, though. If
                               True, every element of the weight vectors is sampled using the Normal distribution
                               reparameterization trick. Thus, this operation is differentiable.
        :return: weight samples, means and stds
        """
        # adj size = (B,T,T) where B=batch size, T=self.n_nodes
        assert adj.shape[1] == self.n_nodes and adj.shape[2] == self.n_nodes
        means = utils.vec_to_adj_mat(self.means, self.n_nodes)
        # means size = (T,T) where T=self.n_nodes
        means = means.expand(adj.shape[0], self.n_nodes, self.n_nodes)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        stds = torch.ones_like(means)
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        if reparametrized:
            samples = torch.distributions.normal.Normal(means, stds).rsample()
            # samples size = (B,T,T) where B=batch size, T=self.n_nodes
        else:
            samples = torch.distributions.normal.Normal(means, stds).sample()
            # samples size = (B,T,T) where B=batch size, T=self.n_nodes

        samples = samples * adj
        # samples size = (B,T,T) where B=batch size, T=self.n_nodes
        means = means * adj
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        stds = stds * adj
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        return samples, means, stds

    def mode(self, adj, return_stds=False):
        """
        Compute a batch of weights modes given a batch of adjacency matrices
        :param adj: batch of adjacency matrices
        :param return_stds: if True, the stds parameters are returned as well
        :return: batch of weights modes or batch of weight modes, stds
        """
        # adj size = (B,T,T) where B=batch size, T=self.n_nodes
        assert [adj.shape[1], adj.shape[2]] == [self.n_nodes] * 2
        means = utils.vec_to_adj_mat(self.means, self.n_nodes)
        # means size = (T,T) where T=self.n_nodes
        means = means.expand(adj.shape[0], self.n_nodes, self.n_nodes)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        means = means * adj
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        stds = torch.ones_like(means)
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        stds = stds * adj
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        if return_stds:
            return means, stds
        else:
            return means



