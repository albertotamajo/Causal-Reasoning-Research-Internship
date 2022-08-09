import torch
import utils


class PriorAdjMatrixDistribution:
    """
    Prior adjacency matrix distribution. The distribution over graph structures is a Gibbs distribution as proposed in
    NOTEARS.
    """
    def __init__(self, n_nodes):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        """
        super().__init__()
        self.n_nodes = n_nodes

    def log_prob(self, adj_matrices, gibbs_temp=10.0, sparsity_factor=0.0):
        """
        Compute the log probability of the given batch of adjacency matrices
        :param adj_matrices: batch of adjacency matrices
        :param gibbs_temp: non-negative scalar temperature for the DAG constraint
        :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
        :return: log probability for the batch of adjacency matrices
        """
        # adj_matrices size = (B,T,T) where B=batch size, T=self.n_nodes
        assert [adj_matrices.shape[1], adj_matrices.shape[2]] == [self.n_nodes] * 2
        dagness = utils.expm(adj_matrices, self.n_nodes).unsqueeze(1)
        # dagness size = (B,1) where B=batch size
        sparsity = torch.sum(adj_matrices, dim=[-1, -2]).unsqueeze(1)
        # sparsity size = (B,1) where B=batch size
        adj_log_probs = - gibbs_temp * dagness - sparsity_factor * sparsity
        # adj_log_probs size = (B,1) where B=batch size
        return adj_log_probs


class PriorWeightsDistribution:
    """
    Prior structural equation parameters distribution. The conditional distribution over structural equation parameters
    given an adjacency matrix is the one proposed in my paper.
    """
    def __init__(self, n_nodes):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        """
        super().__init__()
        self.n_nodes = n_nodes

    def log_prob(self, adj_matrices, weights):
        """
        Compute the log probability of the given batch of weights given a batch of adjacency matrices.
        This function assumes that the weight for a non-parent node is 0 in the provided batch of weights.
        :param adj_matrices: batch of adjacency matrices
        :param weights: batch of weights
        :return: log probability for the batch of weights
        """
        # adj_matrices size = (B,T,T) where B=batch size, T=self.n_nodes
        # weights size = (B,T,T) where B=batch size, T=self.n_nodes
        assert [adj_matrices.shape[1], adj_matrices.shape[2]] == [self.n_nodes] * 2 \
               and [weights.shape[1], weights.shape[2]] == [self.n_nodes] * 2
        means = torch.zeros_like(weights)
        # means size = (B,T,T) where B=batch size, T=self.n_nodes
        stds = torch.ones_like(weights)
        # stds size = (B,T,T) where B=batch size, T=self.n_nodes
        weight_log_probs = torch.distributions.normal.Normal(means, stds).log_prob(weights) * adj_matrices
        # weight_log_probs size = (B,T,T) where B=batch size, T=self.n_nodes
        weight_log_probs = torch.sum(weight_log_probs, dim=(2, 1)).unsqueeze(1)
        # weight_log_probs size = (B,1) where B=batch size
        return weight_log_probs


class PriorLen2Cycles:  #TODO: IT MIGHT BE BROKEN
    """
    Prior adjacency matrix distribution. Actually, this is more of a DAG regularisation but for consistency with
    the rest of the code, it is treated as a distribution. It penalises length-2 cycles in a given adjacency matrix
    factorised distribution.
    """
    def __init__(self, n_nodes):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        """
        super().__init__()
        self.n_nodes = n_nodes

    def log_prob(self, probs, lam):
        """
        Given a batch of Bernoulli parameters describing the probability of there being a directed edge between two
        nodes in an adjacency matrix, compute the likelihood of the given batch of adjacency matrix factorised
        distributions with respect to a prior that penalises length-2 cycles.
        :param probs: batch of Bernoulli parameters describing adjacency matrix factorised distributions
        :param lam: non-negative scalar temperature for the penalisation of length-2 cycles
        :return: penalisation term
        """
        assert [probs.shape[1], probs.shape[2]] == [self.n_nodes] * 2
        # probs size = (B,T,T) where B=batch size, T=n_nodes
        probs_T = torch.transpose(probs, dim0=1, dim1=2)
        # probs_T size = (B,T,T) where B=batch size, T=n_nodes
        reg = torch.sum(torch.cosh(probs * probs_T), dim=(2, 1)).unsqueeze(1)
        # reg size = (B,1) where B=batch size
        return -(lam * reg)


class PriorJointDistribution:
    """
    Prior adjacency matrix and structural equation parameters joint distribution.
    The distribution over graph structures is a Gibbs distribution as proposed in
    NOTEARS. The conditional distribution over structural equation parameters given
    an adjacency matrix is the one proposed in my paper.
    """
    def __init__(self, n_nodes):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.prior_adj_dist = PriorAdjMatrixDistribution(self.n_nodes)
        self.prior_weights_dist = PriorWeightsDistribution(self.n_nodes)

    def log_prob(self, adj_matrices, weights, gibbs_temp=10.0, sparsity_factor=0.0):
        """
        Compute the log probability of the given batch of adjacency matrices and weights.
        This function assumes that the weight for a non-parent node is 0 in the provided batch of weights.
        :param adj_matrices: batch of adjacency matrices
        :param weights: batch of weights
        :param gibbs_temp: non-negative scalar temperature for the DAG constraint
        :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
        :return: log probability for the batch of adjacency matrices and weights
        """
        # adj_matrices size = (B,T,T) where B=batch size, T=self.n_nodes
        # weights size = (B,T,T) where B=batch size, T=self.n_nodes
        assert [adj_matrices.shape[1], adj_matrices.shape[2]] == [self.n_nodes] * 2 \
               and [weights.shape[1], weights.shape[2]] == [self.n_nodes] * 2
        adj_log_probs = self.prior_adj_dist.log_prob(adj_matrices, gibbs_temp, sparsity_factor)
        # adj_log_probs size = (B,1) where B=batch size
        weight_log_probs = self.prior_weights_dist.log_prob(adj_matrices, weights)
        # weight_log_probs size = (B,1) where B=batch size
        log_probs = adj_log_probs + weight_log_probs
        # log_probs size = (B,1) where B=batch size
        return log_probs