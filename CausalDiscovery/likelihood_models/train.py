import numpy as np
from lscm import LSCM
import torch
#from score_functions import bic
import utils
from distributions.data import BIC
from distributions.prior import PriorLen2Cycles


def apply_random_intervention_to_node(scm, n):
    """
    Apply a random intervention to a given node of a Structural Causal Model.The intervention is a hard intervention as
    the value of the intervened node is fixed to a certain value randomly drawn from a standard normal distribution.
    :param scm: Structural Causal Model
    :param n: index of node upon which the intervention is performed
    :return: interventional Structural Causal Model
    """
    return scm.do(n, lambda n_samples: np.repeat(np.random.randn(1), n_samples))


def apply_random_intervention(scm):
    """
    Apply a random intervention to a given Structural Causal Model. The node upon which the intervention is made is
    picked uniformly. The intervention is a hard intervention as the value of the intervened node is fixed to a
    certain value randomly drawn from a standard normal distribution.
    :param scm: Structural Causal Model
    :return: interventional Structural Causal Model, index of intervened node
    """
    n = np.random.randint(len(scm))
    return scm.do(n, lambda n_samples: np.repeat(np.random.randn(1), n_samples)), n


def gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp):
    """
    Given total number of epochs, initial and maximum Gibbs temperature, return a function updating Gibbs temperature
    given its current value and current epoch
    :param tot_epochs: total number of epochs used for the training procedure
    :param init_gibbs_temp: initial Gibbs temperature
    :param max_gibbs_temp: maximum Gibbs temperature
    :return: function updating Gibbs temperature given its current value and current epoch
    """
    def _gibbs_update(curr, epoch):
        """
        Update Gibbs temperature given its current value and current epoch
        :param curr: current Gibbs temperature
        :param epoch: current epoch
        :return: updated Gibbs temperature
        """
        if epoch < tot_epochs * 0.05:
            return curr
        else:
            return init_gibbs_temp + (max_gibbs_temp - init_gibbs_temp) * (
                    10 ** (-2 * max(0, (tot_epochs - 1.1 * epoch) / tot_epochs)))
    return _gibbs_update


def train1(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, max_gibbs_temp, gibbs_update, sparsity_factor,
           n_adj_data_samples, n_weight_data_samples, n_adj_joint_dist_samples, n_weight_joint_dist_samples,
           tot_epochs, weight_dist_update_steps, adj_dist_update_steps, n_interventions, predict_intervention,
           intervention_scoring_steps, adj_dist_optimizer, weight_dist_optimizer):
    """
    Structural Causal Model discovery algorithm. This algorithm takes inspiration from the
    "Learning Neural Causal Models from Unknown Interventions" paper. It does not use REINFORCE though.
    TODO: this algorithm's pseudocode needs to be written
    :param scm: underlying real Structural Causal Model
    :param joint_dist: adjacency matrix and structural equation parameters joint distribution
    :param prior_adj_dist: prior adjacency matrix distribution
    :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
                      parameters
    :param init_gibbs_temp: initial non-negative scalar temperature for the DAG constraint. A value always needs to be
                            set.
    :param max_gibbs_temp: maximum non-negative scalar temperature for the DAG constraint. It can be None if the
                           argument to the gibbs_update parameter is False.
    :param gibbs_update: if True, the Gibbs temperature will be updated during the training procedure following the
                         annealing schedule of the gibbs_anneal function
    :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
    :param n_adj_data_samples: number of samples to be drawn from the intervened Structural Causal Model when updating
                               the adjacency matrix distribution
    :param n_weight_data_samples: number of samples to be drawn from the real Structural Causal Model when updating the
                                  structural equation parameters distribution
    :param n_adj_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution when updating the adjacency matrix distribution
    :param n_weight_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                        parameters joint distribution when updating the structural equation parameters
                                        distribution
    :param weight_dist_update_steps: number of steps to be used for the update of the structural equation parameters
                                     distribution
    :param adj_dist_update_steps: number of steps to be used for the update of the adjacency matrix distribution
    :param n_interventions: number of interventions to be performed within each training epoch
    :param predict_intervention: if True, the training procedure adopts a heuristic algorithm to try predict the
                                 intervened node. If False, it is assumed that the intervened node is known a priori.
    :param intervention_scoring_steps: # TODO when implementing the code to predict interventions, check whether this
                                            parameter is still useful
    :param adj_dist_optimizer: adjacency matrix distribution optimizer
    :param weight_dist_optimizer: structural equation parameters distribution optimizer
    :return: None
    """
    if gibbs_update:
        gibbs_update = gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp)
    else:
        gibbs_update = None

    weight_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    adj_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    for e in range(tot_epochs):
        for _ in range(weight_dist_update_steps):
            loss, data_likelihood, prior_log_prob = weight_LSCM.fit(weight_dist_optimizer, n_weight_data_samples,
                                                                    n_weight_joint_dist_samples, e,
                                                                    include_prior_in_loss=False)
        for _ in range(n_interventions):
            intervention = apply_random_intervention(scm)
            if predict_intervention:
                # code to predict intervention TODO: implement code to predict interventions
                pass
            for _ in range(adj_dist_update_steps):
                loss, data_likelihood, prior_log_prob = adj_LSCM.fit(adj_dist_optimizer, n_adj_data_samples,
                                                                     n_adj_joint_dist_samples, e,
                                                                     include_prior_in_loss=True,
                                                                     intervention=intervention)
        print("Epoch number ", e)
        print(joint_dist.adj_dist.sample(1, return_probs=True)[1])  #TODO DELETE THESE PRINT LINES


def train2(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, max_gibbs_temp, gibbs_update, sparsity_factor,
           n_adj_data_samples, n_weight_data_samples, n_adj_joint_dist_samples, n_weight_joint_dist_samples,
           tot_epochs, weight_dist_update_steps, adj_dist_update_steps, adj_dist_optimizer, weight_dist_optimizer):
    """
    Structural Causal Model discovery algorithm. This algorithm takes inspiration from the
    "Learning Neural Causal Models from Unknown Interventions" paper. However, it differs from the latter as a single
    intervention for every node is performed when updating the adjacency matrix distribution. After intervening on a
    node, the log likelihood of multiple batches of data is computed. These likelihoods are then averaged out and used
    to compute the gradient for the update of the Bernoulli parameters in the adjacency matrix distribution.
    TODO: this algorithm's pseudocode needs to be written
    :param scm: underlying real Structural Causal Model
    :param joint_dist: adjacency matrix and structural equation parameters joint distribution
    :param prior_adj_dist: prior adjacency matrix distribution
    :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
                      parameters
    :param init_gibbs_temp: initial non-negative scalar temperature for the DAG constraint. A value always needs to be
                            set.
    :param max_gibbs_temp: maximum non-negative scalar temperature for the DAG constraint. It can be None if the
                           argument to the gibbs_update parameter is False.
    :param gibbs_update: if True, the Gibbs temperature will be updated during the training procedure following the
                         annealing schedule of the gibbs_anneal function
    :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
    :param n_adj_data_samples: number of samples to be drawn from the intervened Structural Causal Model when updating
                               the adjacency matrix distribution
    :param n_weight_data_samples: number of samples to be drawn from the real Structural Causal Model when updating the
                                  structural equation parameters distribution
    :param n_adj_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution when updating the adjacency matrix distribution
    :param n_weight_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                        parameters joint distribution when updating the structural equation parameters
                                        distribution
    :param weight_dist_update_steps: number of steps to be used for the update of the structural equation parameters
                                     distribution
    :param adj_dist_update_steps: number of steps to be used for the update of the adjacency matrix distribution
    :param adj_dist_optimizer: adjacency matrix distribution optimizer
    :param weight_dist_optimizer: structural equation parameters distribution optimizer
    :return: None
    """
    if gibbs_update:
        gibbs_update = gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp)
    else:
        gibbs_update = None
    weight_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    adj_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    for e in range(tot_epochs):
        for _ in range(weight_dist_update_steps):
            loss, data_likelihood, prior_log_prob = weight_LSCM.fit(weight_dist_optimizer, n_weight_data_samples,
                                                                    n_weight_joint_dist_samples, e,
                                                                    include_prior_in_loss=False)
        for n in range(len(scm)):
            inter_scm = apply_random_intervention_to_node(scm, n)
            intervention = (inter_scm, n)
            node_losses = []
            adj_dist_optimizer.zero_grad()
            for _ in range(adj_dist_update_steps):
                loss, data_likelihood, prior_log_prob = adj_LSCM.compute_loss(n_adj_data_samples,
                                                                     n_adj_joint_dist_samples, e,
                                                                     include_prior_in_loss=True,
                                                                     intervention=intervention)
                node_losses.append(loss)
            node_loss = torch.mean(torch.stack(node_losses))
            node_loss.backward()
            adj_dist_optimizer.step()
        print("Epoch number ", e)
        print(joint_dist.adj_dist.sample(1, return_probs=True)[1])  #TODO DELETE THESE PRINT LINES


def train3(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, max_gibbs_temp, gibbs_update, sparsity_factor,
           n_adj_data_samples, n_weight_data_samples, n_adj_joint_dist_samples, n_weight_joint_dist_samples,
           tot_epochs, weight_dist_update_steps, adj_dist_update_steps, n_interventions, predict_intervention,
           intervention_scoring_steps, adj_dist_optimizer, weight_dist_optimizer):
    """
    Structural Causal Model discovery algorithm. This algorithm takes inspiration from the
    "Learning Neural Causal Models from Unknown Interventions" paper. It does not use REINFORCE though.
    TODO: this algorithm's pseudocode needs to be written
    :param scm: underlying real Structural Causal Model
    :param joint_dist: adjacency matrix and structural equation parameters joint distribution
    :param prior_adj_dist: prior adjacency matrix distribution
    :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
                      parameters
    :param init_gibbs_temp: initial non-negative scalar temperature for the DAG constraint. A value always needs to be
                            set.
    :param max_gibbs_temp: maximum non-negative scalar temperature for the DAG constraint. It can be None if the
                           argument to the gibbs_update parameter is False.
    :param gibbs_update: if True, the Gibbs temperature will be updated during the training procedure following the
                         annealing schedule of the gibbs_anneal function
    :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
    :param n_adj_data_samples: number of samples to be drawn from the intervened Structural Causal Model when updating
                               the adjacency matrix distribution
    :param n_weight_data_samples: number of samples to be drawn from the real Structural Causal Model when updating the
                                  structural equation parameters distribution
    :param n_adj_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution when updating the adjacency matrix distribution
    :param n_weight_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                        parameters joint distribution when updating the structural equation parameters
                                        distribution
    :param weight_dist_update_steps: number of steps to be used for the update of the structural equation parameters
                                     distribution
    :param adj_dist_update_steps: number of steps to be used for the update of the adjacency matrix distribution
    :param n_interventions: number of interventions to be performed within each training epoch
    :param predict_intervention: if True, the training procedure adopts a heuristic algorithm to try predict the
                                 intervened node. If False, it is assumed that the intervened node is known a priori.
    :param intervention_scoring_steps: # TODO when implementing the code to predict interventions, check whether this
                                            parameter is still useful
    :param adj_dist_optimizer: adjacency matrix distribution optimizer
    :param weight_dist_optimizer: structural equation parameters distribution optimizer
    :return: None
    """
    if gibbs_update:
        gibbs_update = gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp)
    else:
        gibbs_update = None

    weight_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    adj_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    for e in range(tot_epochs):
        for _ in range(weight_dist_update_steps):
            loss, data_likelihood, prior_log_prob = weight_LSCM.fit(weight_dist_optimizer, n_weight_data_samples,
                                                                    n_weight_joint_dist_samples, e,
                                                                    include_prior_in_loss=False)
        for _ in range(adj_dist_update_steps):
            loss, data_likelihood, prior_log_prob = adj_LSCM.fit(adj_dist_optimizer, n_adj_data_samples,
                                                                 n_adj_joint_dist_samples, e,
                                                                 include_prior_in_loss=True)
        print("Epoch number ", e)
        print(joint_dist.adj_dist.sample(1, return_probs=True)[1])  #TODO DELETE THESE PRINT LINES


def train4(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, max_gibbs_temp, gibbs_update, sparsity_factor,
           n_adj_data_samples, n_weight_data_samples, n_adj_joint_dist_samples, n_weight_joint_dist_samples,
           tot_epochs, weight_dist_update_steps, adj_dist_update_steps, n_interventions, predict_intervention,
           intervention_scoring_steps, adj_dist_optimizer, weight_dist_optimizer):
    """
    Structural Causal Model discovery algorithm. This algorithm takes inspiration from the
    "Learning Neural Causal Models from Unknown Interventions" paper. It does not use REINFORCE though.
    TODO: this algorithm's pseudocode needs to be written
    :param scm: underlying real Structural Causal Model
    :param joint_dist: adjacency matrix and structural equation parameters joint distribution
    :param prior_adj_dist: prior adjacency matrix distribution
    :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
                      parameters
    :param init_gibbs_temp: initial non-negative scalar temperature for the DAG constraint. A value always needs to be
                            set.
    :param max_gibbs_temp: maximum non-negative scalar temperature for the DAG constraint. It can be None if the
                           argument to the gibbs_update parameter is False.
    :param gibbs_update: if True, the Gibbs temperature will be updated during the training procedure following the
                         annealing schedule of the gibbs_anneal function
    :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
    :param n_adj_data_samples: number of samples to be drawn from the intervened Structural Causal Model when updating
                               the adjacency matrix distribution
    :param n_weight_data_samples: number of samples to be drawn from the real Structural Causal Model when updating the
                                  structural equation parameters distribution
    :param n_adj_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution when updating the adjacency matrix distribution
    :param n_weight_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                        parameters joint distribution when updating the structural equation parameters
                                        distribution
    :param weight_dist_update_steps: number of steps to be used for the update of the structural equation parameters
                                     distribution
    :param adj_dist_update_steps: number of steps to be used for the update of the adjacency matrix distribution
    :param n_interventions: number of interventions to be performed within each training epoch
    :param predict_intervention: if True, the training procedure adopts a heuristic algorithm to try predict the
                                 intervened node. If False, it is assumed that the intervened node is known a priori.
    :param intervention_scoring_steps: # TODO when implementing the code to predict interventions, check whether this
                                            parameter is still useful
    :param adj_dist_optimizer: adjacency matrix distribution optimizer
    :param weight_dist_optimizer: structural equation parameters distribution optimizer
    :return: None
    """
    if gibbs_update:
        gibbs_update = gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp)
    else:
        gibbs_update = None

    samples = scm.sample(256)
    weight_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    adj_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
    for _ in range(weight_dist_update_steps):
        loss, data_likelihood, prior_log_prob = weight_LSCM.fit(weight_dist_optimizer, n_weight_data_samples,
                                                                    n_weight_joint_dist_samples, 0,
                                                                    include_prior_in_loss=False)
    for _ in range(adj_dist_update_steps):
        adj_dist_optimizer.zero_grad()
        adj_samples, weight_samples = joint_dist(1)
        weight_samples = weight_samples.squeeze(0)
        print(weight_samples)
        bic_score = bic(weight_samples, samples, 1) + 150 * cycle_constraint(joint_dist.adj_dist)
        bic_score.backward()
        adj_dist_optimizer.step()
        print(joint_dist.adj_dist.sample(1, return_probs=True)[1])  # TODO DELETE THESE PRINT LINES


def train6(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, max_gibbs_temp, gibbs_update, sparsity_factor,
           n_adj_data_samples, n_weight_data_samples, n_adj_joint_dist_samples, n_weight_joint_dist_samples,
           tot_epochs, weight_dist_update_steps, adj_dist_update_steps, n_interventions, predict_intervention,
           intervention_scoring_steps, adj_dist_optimizer, weight_dist_optimizer):
    """
    Structural Causal Model discovery algorithm. This algorithm takes inspiration from the
    "Learning Neural Causal Models from Unknown Interventions" paper. It does not use REINFORCE though.
    TODO: this algorithm's pseudocode needs to be written
    :param scm: underlying real Structural Causal Model
    :param joint_dist: adjacency matrix and structural equation parameters joint distribution
    :param prior_adj_dist: prior adjacency matrix distribution
    :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
                      parameters
    :param init_gibbs_temp: initial non-negative scalar temperature for the DAG constraint. A value always needs to be
                            set.
    :param max_gibbs_temp: maximum non-negative scalar temperature for the DAG constraint. It can be None if the
                           argument to the gibbs_update parameter is False.
    :param gibbs_update: if True, the Gibbs temperature will be updated during the training procedure following the
                         annealing schedule of the gibbs_anneal function
    :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
    :param n_adj_data_samples: number of samples to be drawn from the intervened Structural Causal Model when updating
                               the adjacency matrix distribution
    :param n_weight_data_samples: number of samples to be drawn from the real Structural Causal Model when updating the
                                  structural equation parameters distribution
    :param n_adj_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                     parameters joint distribution when updating the adjacency matrix distribution
    :param n_weight_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                        parameters joint distribution when updating the structural equation parameters
                                        distribution
    :param weight_dist_update_steps: number of steps to be used for the update of the structural equation parameters
                                     distribution
    :param adj_dist_update_steps: number of steps to be used for the update of the adjacency matrix distribution
    :param n_interventions: number of interventions to be performed within each training epoch
    :param predict_intervention: if True, the training procedure adopts a heuristic algorithm to try predict the
                                 intervened node. If False, it is assumed that the intervened node is known a priori.
    :param intervention_scoring_steps: # TODO when implementing the code to predict interventions, check whether this
                                            parameter is still useful
    :param adj_dist_optimizer: adjacency matrix distribution optimizer
    :param weight_dist_optimizer: structural equation parameters distribution optimizer
    :return: None
    """
    if gibbs_update:
        gibbs_update = gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp)
    else:
        gibbs_update = None

    samples = scm.sample(256)
    def weight(adj_samples):
            return adj_samples * torch.tensor([[[0., 0.5], [0.4, 0.]]]).to(device=adj_samples.device())

    for _ in range(adj_dist_update_steps):
            adj_dist_optimizer.zero_grad()
            adj_samples = joint_dist.adj_dist(1)
            weight_samples = weight(adj_samples)#.squeeze(0)
            bic_score = -BIC(2).log_prob(weight_samples, samples, 1)
            bic_score = bic_score + 150 * cycle_constraint(joint_dist.adj_dist)
            bic_score.backward()
            adj_dist_optimizer.step()
            print(joint_dist.adj_dist.sample(1, return_probs=True)[1])  #TODO DELETE THESE PRINT LINES


def cycle_constraint(adj_dist):  #TODO: DELETE THIS FUNCTION AFTER REFORMATTING CODE FOR THE BIC VARIANT ALGORITHM
    probs = adj_dist._to_probability().unsqueeze(0)
    # probs size = (1,T,1) where T=self.n_dim_out
    probs = utils.vec_to_adj_mat(probs, 2).squeeze(0)
    probs_transpose = probs.T
    return torch.cosh(probs * probs_transpose).sum()


def train7(data_samples, adj_dist, weight_dist, data_dist, epochs, regression_steps, adj_dist_optimizer, weight_dist_optimizer):
    for e in range(epochs):
        adj_dist_optimizer.zero_grad()
        adj_sample = adj_dist(1)
        for _ in range(regression_steps):
            weight_dist_optimizer.zero_grad()
            adj_sample_clone = torch.clone(adj_sample).detach()
            _, means, _ = weight_dist(adj_sample_clone, return_norm_params=True)
            loss = - ((data_dist.log_prob(means, data_samples)).mean())
            loss.backward()
            weight_dist_optimizer.step()
        _, means, _ = weight_dist(adj_sample, return_norm_params=True)
        means = means.squeeze(0)
        bic_score = bic(means, data_samples, 1) + 150 * cycle_constraint(adj_dist)
        bic_score.backward()
        adj_dist_optimizer.step()
        print(adj_dist.sample(1, return_probs=True)[1])  # TODO DELETE THESE PRINT LINES




#TODO THE FOLLOWING FUNCTIONS SHOULD BE USED TO BUOILD NEW TRAINING ALGORITHMS
        
# def train1(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, max_gibbs_temp, gibbs_update, sparsity_factor,
#            n_adj_data_samples, n_weight_data_samples, n_adj_joint_dist_samples, n_weight_joint_dist_samples,
#            tot_epochs, weight_dist_update_steps, adj_dist_update_steps, n_interventions, predict_intervention,
#            intervention_scoring_steps, adj_dist_optimizer, weight_dist_optimizer):
#     """
#     Structural Causal Model discovery algorithm. This algorithm takes inspiration from the
#     "Learning Neural Causal Models from Unknown Interventions" paper. TODO: this algorithm's pseudocode needs to be
#                                                                         written
#     :param scm: underlying real Structural Causal Model
#     :param joint_dist: adjacency matrix and structural equation parameters joint distribution
#     :param prior_adj_dist: prior adjacency matrix distribution
#     :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
#                       parameters
#     :param init_gibbs_temp: initial non-negative scalar temperature for the DAG constraint. A value always needs to be
#                             set.
#     :param max_gibbs_temp: maximum non-negative scalar temperature for the DAG constraint. It can be None if the
#                            argument to the gibbs_update parameter is False.
#     :param gibbs_update: if True, the Gibbs temperature will be updated during the training procedure following the
#                          annealing schedule of the gibbs_anneal function
#     :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
#     :param n_adj_data_samples: number of samples to be drawn from the intervened Structural Causal Model when updating
#                                the adjacency matrix distribution
#     :param n_weight_data_samples: number of samples to be drawn from the real Structural Causal Model when updating the
#                                   structural equation parameters distribution
#     :param n_adj_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
#                                      parameters joint distribution when updating the adjacency matrix distribution
#     :param n_weight_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
#                                         parameters joint distribution when updating the structural equation parameters
#                                         distribution
#     :param weight_dist_update_steps: number of steps to be used for the update of the structural equation parameters
#                                      distribution
#     :param adj_dist_update_steps: number of steps to be used for the update of the adjacency matrix distribution
#     :param n_interventions: number of interventions to be performed within each training epoch
#     :param predict_intervention: if True, the training procedure adopts a heuristic algorithm to try predict the
#                                  intervened node. If False, it is assumed that the intervened node is known a priori.
#     :param intervention_scoring_steps: # TODO when implementing the code to predict interventions, check whether this
#                                             parameter is still useful
#     :param adj_dist_optimizer: adjacency matrix distribution optimizer
#     :param weight_dist_optimizer: structural equation parameters distribution optimizer
#     :return:
#     """
#     if gibbs_update:
#         gibbs_update = gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp)
#     else:
#         gibbs_update = None
#
#     weight_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
#     adj_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
#     for e in range(tot_epochs):
#         for _ in range(weight_dist_update_steps):
#             loss, data_likelihood, prior_log_prob = weight_LSCM.fit(weight_dist_optimizer, n_weight_data_samples,
#                                                                     n_weight_joint_dist_samples, e,
#                                                                     include_prior_in_loss=False)
#         for _ in range(n_interventions):
#             #adj_dist_optimizer = torch.optim.Adam(joint_dist.adj_dist.parameters(), lr=5e-2)  # TODO SHOULD BE COMMENTED OUT
#             #intervention = apply_random_intervention(scm)  # TODO SHOULD BE COMMENTED OUT
#             inter_scm = apply_random_intervention_to_node(scm, 0)
#             intervention = (inter_scm, 1)
#             print("Intervened node: ", intervention[1])  #TODO: DELETE PRINT LINE
#             if predict_intervention:
#                 # code to predict intervention TODO: implement code to predict interventions
#                 pass
#             for _ in range(adj_dist_update_steps):
#                 loss, data_likelihood, prior_log_prob = adj_LSCM.fit(adj_dist_optimizer, n_adj_data_samples,
#                                                                      n_adj_joint_dist_samples, e,
#                                                                      include_prior_in_loss=True,
#                                                                      intervention=intervention)
#                 print(loss) #TODO DELETE
#         print("Epoch number ", e)
#         print(joint_dist.adj_dist.sample(1, return_probs=True)[1])  #TODO DELETE THESE PRINT LINES
#
#
# def train2(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, max_gibbs_temp, gibbs_update, sparsity_factor,
#            n_adj_data_samples, n_weight_data_samples, n_adj_joint_dist_samples, n_weight_joint_dist_samples,
#            tot_epochs, weight_dist_update_steps, adj_dist_update_steps, adj_dist_optimizer, weight_dist_optimizer):
#     """
#     Structural Causal Model discovery algorithm. This algorithm takes inspiration from the
#     "Learning Neural Causal Models from Unknown Interventions" paper. However, it differs from the latter as a single
#     intervention for every node is performed when updating the adjacency matrix distribution. After intervening on a
#     node, the log likelihood of the data is computed but this won't be used to update the adjacency matrix distribution.
#     Indeed, only after intervening on every node and comouting the log likelihood of data, these likelihoods are summed
#     and this summation is used to update the adjacency matrix distribution. TODO: this algorithm's pseudocode needs to be
#                                                                                 written
#     :param scm: underlying real Structural Causal Model
#     :param joint_dist: adjacency matrix and structural equation parameters joint distribution
#     :param prior_adj_dist: prior adjacency matrix distribution
#     :param data_dist: conditional data distribution given an adjacency matrix and linear structural equation
#                       parameters
#     :param init_gibbs_temp: initial non-negative scalar temperature for the DAG constraint. A value always needs to be
#                             set.
#     :param max_gibbs_temp: maximum non-negative scalar temperature for the DAG constraint. It can be None if the
#                            argument to the gibbs_update parameter is False.
#     :param gibbs_update: if True, the Gibbs temperature will be updated during the training procedure following the
#                          annealing schedule of the gibbs_anneal function
#     :param sparsity_factor: non-negative scalar temperature for the sparsity constraint
#     :param n_adj_data_samples: number of samples to be drawn from the intervened Structural Causal Model when updating
#                                the adjacency matrix distribution
#     :param n_weight_data_samples: number of samples to be drawn from the real Structural Causal Model when updating the
#                                   structural equation parameters distribution
#     :param n_adj_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
#                                      parameters joint distribution when updating the adjacency matrix distribution
#     :param n_weight_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
#                                         parameters joint distribution when updating the structural equation parameters
#                                         distribution
#     :param weight_dist_update_steps: number of steps to be used for the update of the structural equation parameters
#                                      distribution
#     :param adj_dist_update_steps: number of steps to be used for the update of the adjacency matrix distribution
#     :param adj_dist_optimizer: adjacency matrix distribution optimizer
#     :param weight_dist_optimizer: structural equation parameters distribution optimizer
#     :return: #TODO MODIFY DESCRIPTION IN CASE i DO NOT CHANGE ALGORITHM
#     """
#     if gibbs_update:
#         gibbs_update = gibbs_anneal(tot_epochs, init_gibbs_temp, max_gibbs_temp)
#     else:
#         gibbs_update = None
#
#     weight_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
#     adj_LSCM = LSCM(scm, joint_dist, prior_adj_dist, data_dist, init_gibbs_temp, sparsity_factor, gibbs_update)
#     for e in range(tot_epochs):
#         for _ in range(weight_dist_update_steps):
#             loss, data_likelihood, prior_log_prob = weight_LSCM.fit(weight_dist_optimizer, n_weight_data_samples,
#                                                                     n_weight_joint_dist_samples, e,
#                                                                     include_prior_in_loss=False)
#         losses = []
#         for n in range(len(scm)):
#             inter_scm = apply_random_intervention_to_node(scm, n)
#             intervention = (inter_scm, n)
#             node_losses = []
#             adj_dist_optimizer.zero_grad()
#             for _ in range(adj_dist_update_steps):
#                 loss, data_likelihood, prior_log_prob = adj_LSCM.compute_loss(n_adj_data_samples,
#                                                                      n_adj_joint_dist_samples, e,
#                                                                      include_prior_in_loss=True,
#                                                                      intervention=intervention)
#                 node_losses.append(loss)
#             node_loss = torch.mean(torch.stack(node_losses))
#             node_loss.backward()
#             adj_dist_optimizer.step()
#             #losses.append(torch.mean(torch.stack(node_losses)))
#         #adj_dist_optimizer.zero_grad()
#         #print(torch.stack(losses))  #TODO: DELETE PRINT LINE
#         #interv_loss = torch.mean(torch.stack(losses))
#         #interv_loss.backward()
#         adj_dist_optimizer.step()
#
#         print("Epoch number ", e)
#         print(joint_dist.adj_dist.sample(1, return_probs=True)[1])  #TODO DELETE THESE PRINT LINES