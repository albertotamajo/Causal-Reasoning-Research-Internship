import torch.optim
from distributions import data, joint, prior, weight
import distributions.adjacency as adj
import likelihood_models.train as train
import numpy as np
from structures import SCM
#from score_functions import bic

scm = SCM({
    "x0": lambda x1, n_samples: 0.5 * x1 + np.random.randn(n_samples),
    "x1": lambda     n_samples: np.random.randn(n_samples)
}, device="cuda")


adj_dist = adj.AdjMatrixFactorisedDistribution(2)
weight_dist = weight.LinWeightsDistribution(2, weight.MLPs_NodeLinWeightsDistribution)
joint_dist = joint.Factorised_JointDistribution(adj_dist, weight_dist, 2)
data_dist = data.LinearDataDistribution(2)
joint_dist.to("cuda")
# TODO CHECK ARGUMENTS AS THEY MIGHT BE INCORRECT
# train.train1(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(2), data_dist=data_dist,
#              init_gibbs_temp=0.0, max_gibbs_temp=50, gibbs_update=False, sparsity_factor=0.0, n_adj_data_samples=256,
#              n_weight_data_samples=256, n_adj_joint_dist_samples=1000, n_weight_joint_dist_samples=1, tot_epochs=2000,
#              weight_dist_update_steps=5000, adj_dist_update_steps=10, n_interventions=1, predict_intervention=False,
#              intervention_scoring_steps=0, adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
#              weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))

# train.train3(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(2), data_dist=data_dist,
#              init_gibbs_temp=0.0, max_gibbs_temp=50, gibbs_update=False, sparsity_factor=0.0, n_adj_data_samples=256,
#              n_weight_data_samples=256, n_adj_joint_dist_samples=1, n_weight_joint_dist_samples=1, tot_epochs=2000,
#              weight_dist_update_steps=5000, adj_dist_update_steps=10, n_interventions=1, predict_intervention=False,
#              intervention_scoring_steps=0, adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
#              weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))

# train.train2(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(2), data_dist=data_dist,
#              init_gibbs_temp=0.0, max_gibbs_temp=50, gibbs_update=False, sparsity_factor=0.1, n_adj_data_samples=256,
#              n_weight_data_samples=256, n_adj_joint_dist_samples=1000, n_weight_joint_dist_samples=1, tot_epochs=2000,
#              weight_dist_update_steps=1000, adj_dist_update_steps=20,
#              adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
#              weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))

# real_likelihood = []
# fake_likelihood = []
# for i in range(1000):
#     sample = scm.sample(256)
#     weight = torch.tensor([[[0., 10], [0., 0.]]]).cuda()
#     real_likelihood.append(data_dist.log_prob(weight, sample))
#
#     weight = torch.tensor([[[0., 10], [0.1, 0.]]]).cuda()
#     fake_likelihood.append(data_dist.log_prob(weight, sample))
#
# print("Average real likelihood: ", torch.mean(torch.stack(real_likelihood)))
# print("Average fake likelihood: ", torch.mean(torch.stack(fake_likelihood)))

# real_bic = []
# fake_bic = []
# for i in range(1):
#     sample = scm.sample(256)
#     weight = torch.tensor([[0., 0.5], [0., 0.]]).cuda()
#     real_bic.append(bic(weight, sample, 1.))
#
#     weight = torch.tensor([[0., 0.], [0.4, 0.]]).cuda()
#     fake_bic.append(bic(weight, sample, 1))
#
# print("Average real likelihood: ", torch.mean(torch.stack(real_bic)))
# print("Average fake likelihood: ", torch.mean(torch.stack(fake_bic)))
# train.train4(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(2), data_dist=data_dist,
#              init_gibbs_temp=0.0, max_gibbs_temp=50, gibbs_update=False, sparsity_factor=0.0, n_adj_data_samples=256,
#              n_weight_data_samples=256, n_adj_joint_dist_samples=1, n_weight_joint_dist_samples=1, tot_epochs=2000,
#              weight_dist_update_steps=5000, adj_dist_update_steps=1000, n_interventions=1, predict_intervention=False,
#              intervention_scoring_steps=0, adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
#              weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))

train.train6(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(2), data_dist=data_dist,
             init_gibbs_temp=0.0, max_gibbs_temp=50, gibbs_update=False, sparsity_factor=0.0, n_adj_data_samples=256,
             n_weight_data_samples=256, n_adj_joint_dist_samples=1, n_weight_joint_dist_samples=1, tot_epochs=2000,
             weight_dist_update_steps=5000, adj_dist_update_steps=1000, n_interventions=1, predict_intervention=False,
             intervention_scoring_steps=0, adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
             weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))

# sample = scm.sample(2000).cuda()
# train.train7(sample, adj_dist, weight_dist, data_dist, 200, 500,
#              adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
#              weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))

