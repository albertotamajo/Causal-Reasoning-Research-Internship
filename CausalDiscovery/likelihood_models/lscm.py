import torch.nn as nn


class LSCM(nn.Module):
    """
    Likelihood Structural Causal Model fitting algorithm
    """
    def __init__(self, scm, joint_dist, prior_adj_dist, data_dist, gibbs_temp=10., sparsity_factor=0.0,
                 gibbs_update=None):
        """
        Initialise algorithm
        :param scm: underlying real Structural Causal Model
        :param joint_dist:  adjacency matrix and structural equation parameters joint distribution
        :param prior_adj_dist: prior adjacency matrix distribution
        :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
                          parameters
        :param gibbs_temp: non-negative scalar temperature for the DAG constraint
        :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
        :param gibbs_update: schedule function for annealing the gibbs temperature
        """
        super().__init__()
        assert len(scm) == joint_dist.n_nodes
        self.scm = scm
        self.joint_dist = joint_dist
        self.prior_adj_dist = prior_adj_dist
        self.data_dist = data_dist
        self.gibbs_temp = gibbs_temp
        self.sparsity_factor = sparsity_factor
        self.gibbs_update = gibbs_update

    def forward(self, n_data_samples, n_joint_dist_samples, e, intervention=None):
        """
        Compute the log likelihood of data with respect to the joint distribution and the log likelihood of the
        joint distribution's samples with respect to the prior adjacency matrix distribution by sampling
        a batch of data samples from the real Structural Causal Model and a batch of samples from the joint
        distribution.
        :param n_data_samples: number of samples to be drawn from the real Structural Causal Model
        :param n_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution
        :param e: epoch number
        :param intervention: if None, the data samples are drawn from the underlying real Structural Causal Model.
                             If not None, a pair needs to be provided where the first element is the interventional
                             Structural Causal Model and the second element is the index of the node subject to the
                             intervention. The computation of the log-likelihood of the data will not involve the
                             intervened node as described in the paper "Learning Neural Causal Models from Unknown
                             Interventions"
        :return: log likelihood of data with respect to the joint distribution , log likelihood of the
                 joint distribution's samples with respect to the prior adjacency matrix distribution
        """
        adj_samples, weight_samples = self.joint_dist(n_joint_dist_samples)
        # adj_samples size = (L,T,T) where L=n_joint_dist_samples, T=n_nodes
        # weight_samples size = (L,T,T) where L=n_joint_dist_samples, T=n_nodes
        self.gibbs_temp = self._update_gibbs_temp(e)
        prior_log_probs = self.prior_adj_dist.log_prob(adj_samples, self.gibbs_temp, self.sparsity_factor)
        # prior_log_probs size = (L,1) where L=n_joint_dist_samples
        if intervention is None:
            data_samples = self.scm.sample(n_data_samples)
            # data_samples size = (B,T) where B=n_data_samples, T=n_nodes
            data_likelihood = self.data_dist.log_prob(weight_samples, data_samples)
            # data_likelihood size = (L,1) where L=n_joint_dist_samples
        else:
            interv_scm = intervention[0]
            interv_node = intervention[1]
            data_samples = interv_scm.sample(n_data_samples)
            # data_samples size = (B,T) where B=n_data_samples, T=n_nodes
            data_likelihood = self.data_dist.log_prob(weight_samples, data_samples, interv_node=interv_node)
            # data_likelihood size = (L,1) where L=n_joint_dist_samples
        return data_likelihood, prior_log_probs

    def compute_loss(self, n_data_samples, n_joint_dist_samples, e, include_prior_in_loss=True, intervention=None):
        """
        Compute the negative log-likelihood of the data with respect to the adjacency matrices and structural equation
        parameters drawn from the adjacency matrix and structural equation parameters joint distribution
        :param n_data_samples: number of samples to be drawn from the real Structural Causal Model
        :param n_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution
        :param e: epoch
        :param include_prior_in_loss: if True, the loss will include the the log likelihood of the
                                      joint distribution's samples with respect to the prior adjacency matrix
                                      distribution. Otherwise, the loss will not include it.
        :param intervention: if None, the data samples are drawn from the underlying real Structural Causal Model.
                             If not None, a pair needs to be provided where the first element is the interventional
                             Structural Causal Model and the second element is the index of the node subject to the
                             intervention. The computation of the log-likelihood of the data will not involve the
                             intervened node as described in the paper "Learning Neural Causal Models from Unknown
                             Interventions"
        :return: loss, negative mean log likelihood of data with respect to the joint distribution, negative mean log
                 likelihood of the joint distribution's samples with respect to the prior adjacency matrix distribution
        """
        data_likelihood, prior_log_probs = self(n_data_samples, n_joint_dist_samples, e, intervention)
        # data_likelihood size = (L,1) where L=n_joint_dist_samples
        # prior_log_probs size = (L,1) where L=n_joint_dist_samples
        if include_prior_in_loss is True:
            loss = (-data_likelihood - prior_log_probs).mean()
            # loss size = (1)
        else:
            loss = (-data_likelihood).mean()
            # loss size = (1)
        return loss, (-data_likelihood).mean(), (-prior_log_probs).mean()

    def fit(self, optimizer, n_data_samples, n_joint_dist_samples, e, include_prior_in_loss=True,
            zero_grad_except_parent=False, intervention=None):
        """
        Fit adjacency matrix and structural equation parameters joint distribution to the true posterior.
        :param optimizer: optimizer
        :param n_data_samples: number of samples to be drawn from the real Structural Causal Model
        :param n_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution
        :param e: epoch
        :param include_prior_in_loss: if True, the loss will include the the log likelihood of the
                                      joint distribution's samples with respect to the prior adjacency matrix
                                      distribution. Otherwise, the loss will not include it.
        :param zero_grad_except_parent: if True, all of the Bernoulli parameters in the adjacency matrix distribution
                                        except for those that denote the probability of a node to have the intervened
                                        node node as parent are zeroed out. Thus, this option will only activate when
                                        the intervention argument is not None.
        :param intervention: if None, the data samples are drawn from the underlying real Structural Causal Model.
                             If not None, a pair needs to be provided where the first element is the interventional
                             Structural Causal Model and the second element is the index of the node subject to the
                             intervention. The computation of the log-likelihood of the data will not involve the
                             intervened node as described in the paper "Learning Neural Causal Models from Unknown
                             Interventions"
        :return: loss, negative mean log likelihood of data with respect to the joint distribution, negative mean log
                 likelihood of the joint distribution's samples with respect to the prior adjacency matrix distribution
        """
        optimizer.zero_grad()
        if zero_grad_except_parent is True and intervention is not None:
            hook = self.joint_dist.adj_dist.zero_grad_except_parent(intervention[1])
        loss, data_likelihood, prior_log_probs = self.compute_loss(n_data_samples, n_joint_dist_samples, e,
                                                                   include_prior_in_loss=include_prior_in_loss,
                                                                   intervention=intervention)
        # loss size = (1)
        # data_likelihood = (1)
        # prior_log_probs = (1)
        loss.backward()
        optimizer.step()
        if zero_grad_except_parent is True and intervention is not None:
            hook.remove()
        return loss, data_likelihood, prior_log_probs

    def sample(self, n_samples=10000):
        """
        Sample from the joint distribution
        :param n_samples: number of samples
        :return: adjacency matrix and weight samples
        """
        return self.joint_dist.sample(n_samples)

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
