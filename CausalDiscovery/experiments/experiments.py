from distributions import weight, data, joint
import distributions.adjacency as adj
from structures import SCM
import torch
import numpy as np
import utils

# ####################################BEGINNING EXPERIMENT#######################################
# """
# Experiment LSTM VS Factorised distribution.
# Given an adjacency matrix, the LSTM learns it faster.
# """
# fac = AdjMatrixFactorisedDistribution(4)
# adam = torch.optim.Adam(fac.parameters())
# adj_matrix = torch.tensor([[0.,1.,1.,0.],[1.,0.,0.,0.],[1.,0.,0.,1.],[0.,0.,1.,0.]])
# for i in range(10000):
#     adam.zero_grad()
#     adj = fac(1).squeeze()
#     loss = ((adj_matrix - adj)**2).sum()
#     loss.backward()
#     adam.step()
#
# print(fac.mode())
# print("\n")
# print(adj_matrix)
# print(fac._sample(1)[1])
#
# fac = LSTM_AdjMatrixDistribution(4)
# adam2 = torch.optim.Adam(fac.parameters())
# adj_matrix = torch.tensor([[0.,1.,1.,0.],[1.,0.,0.,0.],[1.,0.,0.,1.],[0.,0.,1.,0.]])
# for i in range(1000):
#     adam2.zero_grad()
#     adj = fac(1).squeeze()
#     loss = ((adj_matrix - adj)**2).sum()
#     loss.backward()
#     adam2.step()
#
# print(fac.mode())
# print("\n")
# print(adj_matrix)
# print(fac._sample(1)[2])
# ####################################END EXPERIMENT#######################################


####################################BEGINNING EXPERIMENT#######################################

"""

"""
scm = SCM({
    "x1": lambda         n_samples: np.random.randn(n_samples),
    "x2": lambda         n_samples: np.random.randn(n_samples),
    "x3": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples),
    "x4": lambda x1, x2, x3, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples)
})

params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
likelihood = data.LinearDataDistribution(4)
adj_matrix = torch.tensor([[[0.,0.,0.,0.],
                            [0.,0.,0.,0.],
                            [1.,1.,0.,0.],
                            [1.,1.,0.,0.]]]).expand(1000, 4, 4)
optimizer1 = torch.optim.Adam(params_dist.parameters(), lr=0.001)
for i in range(1000):
    optimizer1.zero_grad()
    data_samples = scm.sample(100)
    weights = params_dist(adj_matrix)
    loss = - ((likelihood.log_prob(weights, data_samples)).mean())
    loss.backward()
    optimizer1.step()

adj_dist = adj.AdjMatrixFactorisedDistribution(4,  temp_rsample=0.2, init_params=torch.tensor([[0.],[0.],[0.],
                                                                                               [0.],[0.],[0.],
                                                                                               [0.], [0.], [0.],
                                                                                               [0.], [0.], [0.]]))
print(adj_dist.sample(1, return_probs=True))
var_joint_dist = joint.Factorised_JointDistribution(adj_dist, params_dist, 4)
optimizer2 = torch.optim.Adam(adj_dist.parameters(), lr=0.01)
for i in range(10000):
    optimizer2.zero_grad()
    data_samples = scm.sample(100)
    adj_samples, weight_samples = var_joint_dist(1000)
    mean_adj = torch.mean(adj_samples, dim=0)
    mean_weight = torch.mean(weight_samples, dim=0)
    dagness = utils.expm(adj_samples, 4).unsqueeze(1)
    sparsity = torch.sum(adj_samples, dim=[-1, -2]).unsqueeze(1)
    adj_log_probs = - 50. * dagness - 0.01 * sparsity  #TODO IT WAS -20
    loss = (-likelihood.log_prob(weight_samples, data_samples) - adj_log_probs).mean()
    real_weigths = torch.tensor([[[0.,0.,0., 0.], [0.,0.,0.,0.], [0.5, -0.7, 0.0, 0.0], [0.5, -0.7, 0.0, 0.0]]]).expand(1000, 4,4)
    dagness = utils.expm(adj_matrix, 4).unsqueeze(1)
    sparsity = torch.sum(adj_matrix, dim=[-1, -2]).unsqueeze(1)
    adj_log_probs = - 20. * dagness - 0.01 * sparsity
    loss1 = (-likelihood.log_prob(real_weigths, data_samples) - adj_log_probs).mean()
    loss.backward()
    optimizer2.step()
    print("\n")
    print(adj_dist.sample(1, return_probs=True)[1])
    print("Loss: ", loss)
    print("Real loss: ", loss1)
    if i==9999:
        print("\n")
        print(mean_adj)
        print(mean_weight)
        print(loss)
        print(loss1)
####################################END EXPERIMENT#######################################

####################################BEGINNING EXPERIMENT#######################################
# """
#
# """
# scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples),
#     "x4": lambda x1, x2, x3, n_samples: 0.5*x1 - 0.7*x2 +np.random.randn(n_samples)
# })
#
# params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
# likelihood = data.LinearDataDistribution(4)
# adj_matrix = torch.tensor([[[0.,0.,0.,0.],
#                             [0.,0.,0.,0.],
#                             [1.,1.,0.,0.],
#                             [1.,1.,0.,0.]]]).expand(1000, 4, 4)
# optimizer1 = torch.optim.Adam(params_dist.parameters(), lr=0.001)
# for i in range(1000):
#     optimizer1.zero_grad()
#     data_samples = scm.sample(100)
#     weights = params_dist(adj_matrix)
#     loss = - ((likelihood.log_prob(weights, data_samples)).mean())
#     loss.backward()
#     optimizer1.step()
#
# adj_dist = adj.LSTM_AdjMatrixDistribution(4,  temp_rsample=1.0)
# print(adj_dist.sample(1, return_states=True)[2])
# var_joint_dist = joint.LSTM_JointDistribution(adj_dist, params_dist, 4)
# optimizer2 = torch.optim.Adam(adj_dist.parameters(), lr=0.001)
# for i in range(10000):
#     optimizer2.zero_grad()
#     data_samples = scm.sample(100)
#     adj_samples, weight_samples = var_joint_dist(1000)
#     mean_adj = torch.mean(adj_samples, dim=0)
#     mean_weight = torch.mean(weight_samples, dim=0)
#     dagness = utils.expm(adj_samples, 4).unsqueeze(1)
#     sparsity = torch.sum(adj_samples, dim=[-1, -2]).unsqueeze(1)
#     adj_log_probs = - 50. * dagness - 0.01 * sparsity  #TODO IT WAS -20
#     loss = (-likelihood.log_prob(weight_samples, data_samples) - adj_log_probs).mean()
#     loss.backward()
#     optimizer2.step()
#     print("\n")
#     print(adj_dist.sample(1, return_states=True)[2])
#     print("Loss: ", loss)
#     if i==9999:
#         print("\n")
#         print(mean_adj)
#         print(mean_weight)
#         print(loss)
####################################END EXPERIMENT#######################################


####################################BEGINNING EXPERIMENT#######################################
# scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples),
#     "x4": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples)
# })
#
# real_adj_matrix = torch.tensor([[[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]])
# fake_adj_matrix = torch.tensor([[[0.,0.,0.,0.], [0.,0.,1.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]])
# real_weights = torch.tensor([[[0., 0., 0., 0.], [0., 0., 0., 0.], [0.5, -0.7, 0.0, 0.0], [0.5, -0.7, 0.0, 0.0]]])
# fake_weights = torch.tensor([[[0., 0., 0., 0.], [0., 0., -0.2232, 0.], [0.5020, -0.6897, 0.0, 0.0], [0.5027, -0.7, 0.0, 0.0]]])
# likelihood = data.LinearDataDistribution(4)
# real_likelihoods = []
# fake_likelihoods = []
# for i in range(10000):
#     data_samples = scm.sample(100)
#     real_likelihood = -likelihood.log_prob(real_weights, data_samples).item()
#     fake_likelihood = -likelihood.log_prob(fake_weights, data_samples).item()
#     real_likelihoods.append(real_likelihood)
#     fake_likelihoods.append(fake_likelihood)
#
# print("Correct adjacency matrix likelihood: ", torch.tensor(real_likelihoods).mean())
# print("Incorrect adjacency matrix likelihood: ", torch.tensor(fake_likelihoods).mean())
####################################END EXPERIMENT#######################################

####################################BEGINNING EXPERIMENT#######################################
# """
#
# """
# scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples),
#     "x4": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples)
# })
#
# params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
# likelihood = data.LinearDataDistribution(4)
# adj_matrix = torch.tensor([[[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]]).expand(1000, 4, 4)
# optimizer1 = torch.optim.Adam(params_dist.parameters(), lr=0.001)
# for i in range(1000):
#     optimizer1.zero_grad()
#     data_samples = scm.sample(100)
#     weights = params_dist(adj_matrix)
#     loss = - ((likelihood.log_prob(weights, data_samples)).mean())
#     loss.backward()
#     optimizer1.step()
#
# adj_dist = adj.LSTM_AdjMatrixDistribution(4)
# var_joint_dist = joint.LSTM_JointDistribution(adj_dist, params_dist, 4)
# optimizer2 = torch.optim.Adam(adj_dist.parameters(), lr=0.01)
# for i in range(1000):
#     optimizer2.zero_grad()
#     data_samples = scm.sample(100)
#     adj_samples, weight_samples = var_joint_dist(1000)
#     mean_adj = torch.mean(adj_samples, dim=0)
#     mean_weight = torch.mean(weight_samples, dim=0)
#     dagness = utils.expm(adj_samples, 4).unsqueeze(1)
#     sparsity = torch.sum(adj_samples, dim=[-1, -2]).unsqueeze(1)
#     adj_log_probs = - 10. * dagness - 0.01 * sparsity
#     loss = (-likelihood.log_prob(weight_samples, data_samples)).mean() #- adj_log_probs).mean()
#     loss.backward()
#     optimizer2.step()
#     print("\n")
#     print(adj_dist.sample(1, return_states=True))
#     if i==999:
#         print("\n")
#         mod_weight_samples = torch.clone(weight_samples)
#         mod_weight_samples[:, : 2] = 0
#         loss1 = - ((likelihood.log_prob(mod_weight_samples, data_samples)).mean())
#         print(mean_adj)
#         print(mean_weight)
####################################END EXPERIMENT#######################################

####################################BEGINNING EXPERIMENT#######################################
# """
#
# """
# scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 0.5*x1 + 0.9*x2 + np.random.randn(n_samples),
#     "x4": lambda x1, x2, n_samples: 0.5*x1 + 0.7*x2 + np.random.randn(n_samples)
# })
#
# params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
# adj_dist = adj.AdjMatrixFactorisedDistribution(4)
# likelihood = data.LinearDataDistribution(4)
# optimizer1 = torch.optim.Adam(params_dist.parameters(), lr=0.001)
# for i in range(3000):
#     optimizer1.zero_grad()
#     data_samples = scm.sample(100)
#     adj_samples = adj_dist(1)
#     weights = params_dist(adj_samples)
#     loss = - ((likelihood.log_prob(weights, data_samples)).mean())
#     loss.backward()
#     optimizer1.step()
#
# print(params_dist(torch.tensor([[[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]])))
#
#
# var_joint_dist = joint.Factorised_JointDistribution(adj_dist, params_dist, 4)
# optimizer2 = torch.optim.Adam(adj_dist.parameters(), lr=0.01)
# for i in range(10000):
#     optimizer2.zero_grad()
#     data_samples = scm.sample(100)
#     adj_samples, weight_samples = var_joint_dist(1)
#     mean_adj = torch.mean(adj_samples, dim=0)
#     mean_weight = torch.mean(weight_samples, dim=0)
#     dagness = utils.expm(adj_samples, 4).unsqueeze(1)
#     sparsity = torch.sum(adj_samples, dim=[-1, -2]).unsqueeze(1)
#     adj_log_probs = - 10. * dagness - 0.01 * sparsity
#     loss = (-likelihood.log_prob(weight_samples, data_samples)).mean() #- adj_log_probs).mean() TODO FIXED
#     loss.backward()
#     optimizer2.step()
#     print("\n")
#     print(adj_dist.sample(1, return_probs=True))
#     if i==999:
#         print("\n")
#         mod_weight_samples = torch.clone(weight_samples)
#         mod_weight_samples[:, : 2] = 0
#         loss1 = - ((likelihood.log_prob(mod_weight_samples, data_samples)).mean())
#         print(mean_adj)
#         print(mean_weight)
####################################END EXPERIMENT#######################################

####################################BEGINNING EXPERIMENT#######################################
#
# """
#
# """
# scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples),
#     "x4": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples)
# })
#
# params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
# adj_dist = adj.AdjMatrixFactorisedDistribution(4)
# var_joint_dist = joint.Factorised_JointDistribution(adj_dist, params_dist, 4)
# likelihood = data.LinearDataDistribution(4)
# optimizer1 = torch.optim.Adam(var_joint_dist.parameters(), lr=0.01)
# for i in range(10000):
#     optimizer1.zero_grad()
#     data_samples = scm.sample(100)
#     adj_samples, weight_samples = var_joint_dist(1000)
#     dagness = utils.expm(adj_samples, 4).unsqueeze(1)
#     sparsity = torch.sum(adj_samples, dim=[-1, -2]).unsqueeze(1)
#     adj_log_probs = - 30. * dagness - 0.01 * sparsity
#     loss = (-likelihood.log_prob(weight_samples, data_samples) - adj_log_probs).mean()
#     loss.backward()
#     optimizer1.step()
#     print(adj_dist.sample(1, return_probs=True)[1])
#     print(params_dist(torch.tensor([[[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]])))

####################################END EXPERIMENT#######################################


####################################BEGINNING EXPERIMENT#######################################
# """
# """
# scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 1.*x1 - 1.5*x2 + np.random.randn(n_samples),
#     "x4": lambda x1,x2,x3, n_samples: 1.*x1 - 1.5*x2 + np.random.randn(n_samples)
# })
#
#
# params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
# likelihood = data.LinearDataDistribution(4)
# adj_matrix = torch.tensor([[[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]]).expand(1000, 4, 4)
# optimizer = torch.optim.Adam(params_dist.parameters(), lr=0.001)
# for i in range(10000):
#     optimizer.zero_grad()
#     data_samples = scm.sample(100)
#     weights = params_dist(adj_matrix)
#     loss = - ((likelihood.log_prob(weights, data_samples)).mean())
#     loss.backward()
#     optimizer.step()
#     print(params_dist.mode(adj_matrix[0].unsqueeze(0), return_stds=True)[0])
#     print(loss)
####################################END EXPERIMENT#######################################

####################################BEGINNING EXPERIMENT#######################################
# """
# """
#scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 1.*x1 - 1.5*x2 + np.random.randn(n_samples),
#     "x4": lambda x1,x2,x3, n_samples: 1.*x1 - 1.5*x2 + np.random.randn(n_samples)
# })
# adj_dist = adj.AdjMatrixFactorisedDistribution(4)
# params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
# var_joint_dist = joint.Factorised_JointDistribution(adj_dist, params_dist, 4)
# vscm = VSCM_Observational(scm, var_joint_dist, prior.PriorDistribution(4), data.LinearDataDistribution(4), sparsity_factor=0.01)
# adam = torch.optim.Adam(var_joint_dist.parameters())
# baseline = 0
# for e in range(10000):
#         for i in range(100):
#             a, b, c, _ = fit(vscm, adam, baseline, 10, 1000, e)
#             print(var_joint_dist.sample(1, return_meta=True)[2])
#             #print(var_joint_dist.sample(1, return_meta=True)[2])
#             #print(a, b, c)
####################################END EXPERIMENT#######################################

####################################BEGINNING EXPERIMENT#######################################
# """
# """
# scm = SCM({
#     "x1": lambda         n_samples: np.random.randn(n_samples),
#     "x2": lambda         n_samples: np.random.randn(n_samples),
#     "x3": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples),
#     "x4": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples)
# })
#
# adj_dist = adj.AdjMatrixFactorisedDistribution(4,  temp_rsample=1.0)
# params_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
# var_joint_dist = joint.Factorised_JointDistribution(adj_dist, params_dist, 4)
# optimizer1 = torch.optim.Adam(params_dist.parameters(), lr=0.001)
# optimizer2 = torch.optim.Adam(adj_dist.parameters(), lr=0.1)
# likelihood = data.LinearDataDistribution(4)
# adj_matrix = torch.tensor([[[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]]).expand(1000, 4, 4)
# for i in range(1000):
#     optimizer1.zero_grad()
#     data_samples = scm.sample(100)
#     weights = params_dist(adj_matrix)
#     loss = - ((likelihood.log_prob(weights, data_samples)).mean())
#     loss.backward()
#     optimizer1.step()
#
# vscm = VSCM_Observational(scm, var_joint_dist, prior.PriorDistribution(4), data.LinearDataDistribution(4), gibbs_temp=20,
#                           sparsity_factor=0.01)
# baseline = 0.
# for i in range(10000):
#     loss, _, _, b = fit(vscm, optimizer2, baseline, 100, 10000, i)
#     baseline = b
#     print("\n")
#     print(adj_dist.sample(1, return_probs=True)[1])
#     print(loss)


####################################END EXPERIMENT#######################################


















