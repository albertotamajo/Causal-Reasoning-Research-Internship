import torch.nn as nn
from distributions.adjacency import AdjMatrixFactorisedDistribution, LSTM_AdjMatrixDistribution


class LSTM_JointDistribution(nn.Module):
    """
    Adjacency matrix and structural equation parameters joint distribution.
    The adjacency matrix distribution needs to be modelled with a LSTM_AdjMatrixDistribution instance.
    """
    def __init__(self, adj_dist, params_dist, n_nodes):
        """
        Initialise distribution
        :param adj_dist: adjacency matrix distribution. It needs to be an instance of the LSTM_AdjMatrixDistribution
                         class
        :param params_dist: conditional structural equation parameters distribution given adjacency matrix
        :param n_nodes: number of nodes in the graph
        """
        assert isinstance(adj_dist, LSTM_AdjMatrixDistribution)
        super().__init__()
        self.adj_dist = adj_dist
        self.params_dist = params_dist
        self.n_nodes = n_nodes
        assert self.adj_dist.n_nodes == self.params_dist.n_nodes and self.adj_dist.n_nodes == self.n_nodes

    def forward(self, batch_size, reparametrized=True, temp=None, return_meta=False, start_state=None,
                init_input=None):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices
        :param batch_size: number of adjacency matrices to be sampled
        :param reparametrized: if False, the sampling process is not differentiable. Otherwise, it is differentiable
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param return_meta: if True, the LSTM states, Bernoulli success probabilities and Normal distribution parameters
                            are returned as well.
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples, weight samples or adjacency matrix samples, weight samples,
        (LSTM states, Bernoulli success probabilities), (means, stds)
        """
        adj_samples, weight_samples, adj_dist_meta, params_dist_meta = self._sample(batch_size=batch_size,
                                                                                    reparametrized=reparametrized,
                                                                                    temp=temp, start_state=start_state,
                                                                                    init_input=init_input)
        if return_meta:
            return adj_samples, weight_samples, adj_dist_meta, params_dist_meta
        return adj_samples, weight_samples

    def sample(self, batch_size, return_meta=False, temp=None, start_state=None, init_input=None):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices. This
        sampling process is not differentiable.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_meta: if True, the LSTM states, Bernoulli success probabilities and Normal distribution parameters
                            are returned as well.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples, weight samples or  adjacency matrix samples, weight samples,
        (LSTM states, Bernoulli success probabilities), (means, stds)
        """
        adj_samples, weight_samples, adj_dist_meta, params_dist_meta = self._sample(batch_size=batch_size, temp=temp,
                                                                                    start_state=start_state,
                                                                                    init_input=init_input)
        if return_meta:
            return adj_samples, weight_samples, adj_dist_meta, params_dist_meta
        return adj_samples, weight_samples

    def rsample(self, batch_size, return_meta=False, temp=None, start_state=None, init_input=None):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices. This
        sampling process is differentiable.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_meta: if True, the LSTM states, Bernoulli success probabilities and Normal distribution parameters
                            are returned as well.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples, weight samples or  adjacency matrix samples, weight samples,
        (LSTM states, Bernoulli success probabilities), (means, stds)
        """
        adj_samples, weight_samples, adj_dist_meta, params_dist_meta = self._sample(batch_size=batch_size,
                                                                                    reparametrized=True, temp=temp,
                                                                                    start_state=start_state,
                                                                                    init_input=init_input)
        if return_meta:
            return adj_samples, weight_samples, adj_dist_meta, params_dist_meta
        return adj_samples, weight_samples

    def _sample(self, batch_size, reparametrized=False, temp=None, start_state=None, init_input=None):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices
        :param batch_size: number of adjacency matrices to be sampled
        :param reparametrized: if False, the sampling process is not differentiable. Otherwise, it is differentiable
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples, weight samples, (LSTM states, Bernoulli success probabilities), (means, stds)
        """
        adj_samples, states, bern_probs = self.adj_dist(batch_size=batch_size, reparametrized=reparametrized, temp=temp,
                                                        return_states=True, start_state=start_state,
                                                        init_input=init_input)
        # adj_samples size = (B,T,T) where B=batch size, T=self.n_nodes
        # states = tuple where both elements have size (B,Q,Z,H) where B=batch size, Q=adj_dist.n_dim_out,
        # Z=adj_dist.n_layers, H=adj_dist.hidden_dim
        # bern_probs size = (B,T,T) where B=batch size, T=self.n_nodes
        weight_samples, means, stds = self.params_dist(adj=adj_samples, reparametrized=reparametrized,
                                                       return_norm_params=True)
        # weight_sample size = (B,T,T) where B=batch size, T=self.n_nodes
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        return adj_samples, weight_samples, (states, bern_probs), (means, stds)

    def log_prob(self, adj_matrices, weights, return_probs=False):
        """
        Compute the log probability of the given batch of adjacency matrices and weights
        :param adj_matrices: batch of adjacency matrices
        :param weights: batch of weights
        :param return_probs: if True, the Bernoulli success probabilities for each entry of the adjacency matrices and
                             the Normal distribution parameters are returned as well
        :return: log probability for the batch of adjacency matrices and weights or log probability for the batch of
                 adjacency matrices and weights, the Bernoulli success probabilities for each entry of the adjacency
                 matrices, the Normal distribution parameters
        """
        # adj_matrices size = (B,T,T) where B=batch size, T=self.n_nodes
        # weights size = (B,T,T) where B=batch size, T=self.n_nodes
        adj_log_probs, bern_probs = self.adj_dist.log_prob(adj_matrices, return_probs=True)
        # adj_log_probs size = (B,1) where B=batch size
        # bern_probs size = (B,T,T) where B=batch size, T=self.n_nodes
        params_log_probs, means, stds = self.params_dist.log_prob(weights, adj_matrices, return_probs=True)
        # params_log_probs size = (B,1)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        log_probs = adj_log_probs + params_log_probs
        # log_probs size = (B,1) where B=batch size
        if return_probs:
            return log_probs, bern_probs, means, stds
        return log_probs

    def mode(self, n_samples=1000, return_meta=False):
        """
        Compute the mode of the adjacency matrix and structural equation parameters joint distribution
        :param n_samples: number of samples to draw from the adjacency matrix distribution from which to compute the
                          mode
        :param return_meta: if True, the log probability associated with the mode and the stds are returned as well
        :return: mode of the joint distribution or mode of the joint distribution, the log probability associated with
                 the mode, the stds
        """
        # TODO the process here might not be correct, Maybe, I should sample from the joint and then pick the one with
        # TODO the highest log probability. Instead, what I am doing here is more of a greedy approach
        adj_mode = self.adj_dist.mode(n_samples=n_samples)
        # adj_mode size = (1,T,T) where T=self.n_nodes
        weight_mode, stds = self.params_dist.mode(adj_mode, return_stds=True)
        # weight_mode size = (1,T,T) where B=batch size, T=self.n_nodes
        # stds size = (1,T,T) where B=batch size, T=self.n_nodes
        if return_meta:
            log_prob = self.log_prob(adj_mode, weight_mode)
            # log_prob size = (1,1)
            return adj_mode, weight_mode, log_prob, stds
        return adj_mode, weight_mode


class Factorised_JointDistribution(nn.Module):
    """
    Adjacency matrix and structural equation parameters joint distribution.
    The adjacency matrix distribution needs to be modelled with an AdjMatrixFactorisedDistribution instance.
    """
    def __init__(self, adj_dist, params_dist, n_nodes):
        """
        Initialise distribution
        :param adj_dist: adjacency matrix distribution. It needs to be an instance of the
                         AdjMatrixFactorisedDistribution class
        :param params_dist: conditional structural equation parameters distribution given adjacency matrix
        :param n_nodes: number of nodes in the graph
        """
        assert isinstance(adj_dist, AdjMatrixFactorisedDistribution)
        super().__init__()
        self.adj_dist = adj_dist
        self.params_dist = params_dist
        self.n_nodes = n_nodes
        assert self.adj_dist.n_nodes == self.params_dist.n_nodes and self.adj_dist.n_nodes == self.n_nodes

    def forward(self, batch_size, reparametrized=True, temp=None, return_meta=False):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices
        :param batch_size: number of adjacency matrices to be sampled
        :param reparametrized: if False, the sampling process is not differentiable. Otherwise, it is differentiable
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param return_meta: if True, the Bernoulli success probabilities and Normal distribution parameters
                            are returned as well.
        :return: adjacency matrix samples, weight samples or adjacency matrix samples, weight samples,
        (Bernoulli success probabilities), (means, stds)
        """
        adj_samples, weight_samples, adj_dist_meta, params_dist_meta = self._sample(batch_size=batch_size,
                                                                                    reparametrized=reparametrized,
                                                                                    temp=temp)
        if return_meta:
            return adj_samples, weight_samples, adj_dist_meta, params_dist_meta
        return adj_samples, weight_samples

    def sample(self, batch_size, return_meta=False, temp=None):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices. This
        sampling process is not differentiable.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_meta: if True, the Bernoulli success probabilities and Normal distribution parameters
                            are returned as well.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :return: adjacency matrix samples, weight samples or  adjacency matrix samples, weight samples,
        (Bernoulli success probabilities), (means, stds)
        """
        adj_samples, weight_samples, adj_dist_meta, params_dist_meta = self._sample(batch_size=batch_size, temp=temp)
        if return_meta:
            return adj_samples, weight_samples, adj_dist_meta, params_dist_meta
        return adj_samples, weight_samples

    def rsample(self, batch_size, return_meta=False, temp=None):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices. This
        sampling process is differentiable.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_meta: if True, the Bernoulli success probabilities and Normal distribution parameters
                            are returned as well.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :return: adjacency matrix samples, weight samples or  adjacency matrix samples, weight samples,
        (Bernoulli success probabilities), (means, stds)
        """
        adj_samples, weight_samples, adj_dist_meta, params_dist_meta = self._sample(batch_size=batch_size,
                                                                                    reparametrized=True, temp=temp)
        if return_meta:
            return adj_samples, weight_samples, adj_dist_meta, params_dist_meta
        return adj_samples, weight_samples

    def _sample(self, batch_size, reparametrized=False, temp=None):
        """
        Sample a batch of adjacency matrices from the adjacency matrix distribution and thereafter sample a batch of
        weights from the structural equation parameters distribution given the sampled batch of adjacency matrices
        :param batch_size: number of adjacency matrices to be sampled
        :param reparametrized: if False, the sampling process is not differentiable. Otherwise, it is differentiable
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :return: adjacency matrix samples, weight samples, (Bernoulli success probabilities), (means, stds)
        """
        adj_samples, bern_probs = self.adj_dist(batch_size=batch_size, reparametrized=reparametrized, temp=temp,
                                                        return_probs=True)
        # adj_samples size = (B,T,T) where B=batch size, T=self.n_nodes
        # bern_probs size = (B,T,T) where B=batch size, T=self.n_nodes
        weight_samples, means, stds = self.params_dist(adj=adj_samples, reparametrized=reparametrized,
                                                       return_norm_params=True)
        # weight_sample size = (B,T,T) where B=batch size, T=self.n_nodes
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        return adj_samples, weight_samples, (bern_probs), (means, stds)

    def log_prob(self, adj_matrices, weights, return_probs=False):
        """
        Compute the log probability of the given batch of adjacency matrices and weights
        :param adj_matrices: batch of adjacency matrices
        :param weights: batch of weights
        :param return_probs: if True, the Bernoulli success probabilities for each entry of the adjacency matrices and
                             the Normal distribution parameters are returned as well
        :return: log probability for the batch of adjacency matrices and weights or log probability for the batch of
                 adjacency matrices and weights, the Bernoulli success probabilities for each entry of the adjacency
                 matrices, the Normal distribution parameters
        """
        # adj_matrices size = (B,T,T) where B=batch size, T=self.n_nodes
        # weights size = (B,T,T) where B=batch size, T=self.n_nodes
        adj_log_probs, bern_probs = self.adj_dist.log_prob(adj_matrices, return_probs=True)
        # adj_log_probs size = (B,1) where B=batch size
        # bern_probs size = (B,T,T) where B=batch size, T=self.n_nodes
        params_log_probs, means, stds = self.params_dist.log_prob(weights, adj_matrices, return_probs=True)
        # params_log_probs size = (B,1)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        log_probs = adj_log_probs + params_log_probs
        # log_probs size = (B,1) where B=batch size
        if return_probs:
            return log_probs, bern_probs, means, stds
        return log_probs

    def mode(self, return_meta=False):
        """
        Compute the mode of the adjacency matrix and structural equation parameters joint distribution
        :param return_meta: if True, the log probability associated with the mode and the stds are returned as well
        :return: mode of the joint distribution or mode of the joint distribution, the log probability associated with
                 the mode, the stds
        """
        # TODO the process here might not be correct, Maybe, I should sample from the joint and then pick the one with
        # TODO the highest log probability. Instead, what I am doing here is more of a greedy approach
        adj_mode = self.adj_dist.mode()
        # adj_mode size = (1,T,T) where T=self.n_nodes
        weight_mode, stds = self.params_dist.mode(adj_mode, return_stds=True)
        # weight_mode size = (1,T,T) where B=batch size, T=self.n_nodes
        # stds size = (1,T,T) where B=batch size, T=self.n_nodes
        if return_meta:
            log_prob = self.log_prob(adj_mode, weight_mode)
            # log_prob size = (1,1)
            return adj_mode, weight_mode, log_prob, stds
        return adj_mode, weight_mode