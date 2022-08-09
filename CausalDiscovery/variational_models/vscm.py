import torch.nn as nn


class VSCM(nn.Module):
    """
    Variational Structural Causal Model fitting algorithm
    """
    def __init__(self, scm, var_joint_dist, prior_joint_dist, data_dist, gibbs_temp=10., sparsity_factor=0.0,
                 gibbs_update=None):
        """
        Initialise algorithm
        :param scm: underlying real Structural Causal Model
        :param var_joint_dist:  adjacency matrix and structural equation parameters variational joint distribution
        :param prior_joint_dist: adjacency matrix and structural equation parameters prior joint distribution
        :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
                          parameters
        :param gibbs_temp: non-negative scalar temperature for the DAG constraint
        :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
        :param gibbs_update: schedule function for annealing the gibbs temperature
        """
        super().__init__()
        assert len(scm) == var_joint_dist.n_nodes
        self.scm = scm
        self.var_joint_dist = var_joint_dist
        self.prior_joint_dist = prior_joint_dist
        self.data_dist = data_dist
        self.gibbs_temp = gibbs_temp
        self.sparsity_factor = sparsity_factor
        self.gibbs_update = gibbs_update

    def forward(self, n_data_samples, n_joint_dist_samples, e):
        """
        Compute the log likelihood of data, the Kullback-leibler divergence between variational joint distribution and
        prior distribution and the log probability of samples drawn from the variational joint distribution by sampling
        a batch of data samples from the real Structural Causal Model and a batch of samples from the variational joint
        distribution.
        :param n_data_samples: number of samples to be drawn from the real Structural Causal Model
        :param n_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters variational joint distribution
        :param e: epoch number
        :return: log likelihood of data, Kullback-leibler divergence between variational
                 joint distribution and prior distribution, log probability of samples drawn from the variational
                 joint distribution
        """
        data_samples = self.scm.sample(n_data_samples)
        # data_samples size = (B,T) where B=n_data_samples, T=n_nodes
        adj_samples, weight_samples = self.var_joint_dist(n_joint_dist_samples)
        # adj_samples size = (L,T,T) where L=n_joint_dist_samples, T=n_nodes
        # weight_samples size = (L,T,T) where L=n_joint_dist_samples, T=n_nodes
        data_likelihood = self.data_dist.log_prob(weight_samples, data_samples)
        # data_likelihood size = (L,1) where L=n_joint_dist_samples
        var_log_probs = self.var_joint_dist.log_prob(adj_samples, weight_samples)
        # var_log_probs size = (L,1) where L=n_joint_dist_samples
        self.gibbs_temp = self._update_gibbs_temp(e)
        prior_log_probs = self.prior_joint_dist.log_prob(adj_samples, weight_samples, self.gibbs_temp,
                                                         self.sparsity_factor)
        # prior_log_probs size = (L,1) where L=n_joint_dist_samples
        kl = var_log_probs - prior_log_probs
        # kl size = (L,1) where L=n_joint_dist_samples
        return data_likelihood, kl, var_log_probs

    def sample(self, n_samples=10000):
        """
        Sample from the variational joint distribution
        :param n_samples: number of samples
        :return: adjacency matrix and weight samples
        """
        return self.var_joint_dist.sample(n_samples)

    def _update_gibbs_temp(self, e):
        """
        Update Gibbs temperature
        :param e: epoch
        :return: updated Gibbs temperature
        """
        if self.gibbs_update is not None:
            return self.gibbs_update(self.gibbs_temp, e)
        else:
            return self.gibbs_temp