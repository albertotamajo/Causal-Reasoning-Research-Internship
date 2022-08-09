import torch
import math
from functorch import vmap #TODO: functorch is a beta library.


class LinearDataDistribution:
    """
    Conditional data distribution given an adjacency matrix and linear structural equation parameters.
    The underlying assumption is that the value of a node is a random variable whose value depends on a
    linear combination of the value of its parents and an additive standard normal error.
    """

    def __init__(self, n_nodes):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        """
        super().__init__()
        self.n_nodes = n_nodes

    def log_prob(self, weights, data, interv_node=None):
        """
        Compute the log probability of the given batch of data given the batch of weights
        :param weights: batch of weights
        :param data: batch of data
        :param interv_node: if None, the log probability of the given batch of data given the batch of weights is
                            computed. Otherwise, the the computation of the log-likelihood of the data will not involve
                            the intervened node as described in the paper "Learning Neural Causal Models from Unknown
                            Interventions". Thus, when not None, the argument to the interv_node parameter needs to be
                            the index of the intervened node.
        :return: log probability of the given batch of data
        """
        # weights size = (B,T,T) where B=batch size, T=self.n_nodes
        # data size = (L,T) where L=data samples, T=self.n_nodes
        assert [weights.shape[1], weights.shape[2]] == [self.n_nodes] * 2 and data.shape[1] == self.n_nodes
        data = torch.transpose(data, 0, 1)
        # data size = (T,L) where T=self.n_nodes, L=data samples
        if interv_node is not None:
            weights = torch.cat((weights[:, 0:interv_node, :], weights[:, interv_node + 1:, :]), dim=1)
            # weights size = (B,T-1,T) where B=batch size, T=self.n_nodes
        means = torch.matmul(weights, data)
        # means size = (B,T,L) or (B,T-1,L) where B=batch size, T=self.n_nodes, L=data samples
        means = torch.transpose(means, 1, 2)
        # means size = (B,L,T) or (B,L,T-1) where B=batch size, L=data samples, T=self.n_nodes
        stds = torch.ones_like(means)
        # stds size = (B,L,T) or (B,L,T-1) where B=batch size, L=data samples, T=self.n_nodes
        data = torch.transpose(data, 0, 1)
        # data size = (L,T) where L=data samples, T=self.n_nodes
        if interv_node is not None:
            data = torch.cat((data[:, 0:interv_node], data[:, interv_node + 1:]), dim=1)
            # data size = (L,T-1) where L=data samples, T=self.n_nodes
        log_probs = torch.distributions.normal.Normal(means, stds).log_prob(data)
        # log_probs size = (B,L,T) or (B,L,T-1) where B=batch size, L=data samples, T=self.n_nodes
        log_probs = torch.sum(log_probs, dim=(2, 1)).unsqueeze(1)
        # log_probs size = (B,1) where B=batch size
        return log_probs


class BIC:
    """
    Conditional data distribution given an adjacency matrix and linear structural equation parameters.
    The underlying assumption is that the value of a node is a random variable whose value depends on a
    linear combination of the value of its parents and an additive normal error. This distribution allows
    computing the penalised log-likelihood of the data given an adjacency matrix and linear structural equation
    parameters as described in "Identifiability of Gaussian structural equation models with equal error variances".
    Since lambda is chosen to be equal to log(n)/2, minimising the negative penalised log-like of the data is equivalent
    to minimising the BIC score
    """

    def __init__(self, n_nodes):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        """
        super().__init__()
        self.n_nodes = n_nodes

    def log_prob(self, weights, data, var):
        """
        Compute the penalised log-likelihood of the given batch of data given the batch of weights
        :param weights: batch of weights
        :param data: batch of data
        :param var: common variance of the nodes' independent additive normal errors in the graph
        :return: penalised log-likelihood of data
        # TODO: this function uses vmap. Need to check whether vmap actually backpropagates the gradients
        """
        # weights size = (B,T,T) where B=batch size, T=self.n_nodes
        # data size = (L,T) where B = data samples, T=self.n_nodes
        assert [weights.shape[1], weights.shape[2]] == [self.n_nodes] * 2 and data.shape[1] == self.n_nodes
        L = data.shape[0]
        sample_covariance = torch.cov(data.T)
        # sample covariance size = (T,T) where T=self.n_nodes
        identity = torch.eye(self.n_nodes).to(device=weights.device)
        # identity size = (T,T) where T=self.n_nodes
        trace_arg = torch.matmul(torch.matmul(torch.transpose((identity - weights), dim0=1, dim1=2),
                                              (identity - weights)), sample_covariance)
        # trace arg size = (B,T,T) where B=batch size, T=self.n_nodes
        lam = torch.log(torch.tensor(L)) / 2
        non_zeros = torch.sum(torch.count_nonzero(weights, dim=2), dim=1).unsqueeze(1)
        #non_zeros = torch.sum(torch.abs(weights), dim=(2, 1)).unsqueeze(1) #TODO: L1 APPROXIMATION OF L0 NORM
        # non_zeros size = (B,1) where B=batch size
        trace = vmap(torch.trace)(trace_arg).unsqueeze(1)
        # trace size = (B,1) where B=batch size
        bic = ((L * self.n_nodes) / 2) * torch.log(2 * torch.tensor(math.pi) * var) + (L / (2 * var)) * trace + \
            lam * non_zeros
        # bic size = (B,1) where B=batch size
        return -bic
