import torch.optim

from distributions import data, joint, prior, weight
import distributions.adjacency as adj
import train
import numpy as np
from structures import SCM

scm = SCM({
    "x0": lambda         n_samples: np.random.randn(n_samples),
    "x1": lambda         n_samples: np.random.randn(n_samples),
    "x2": lambda x0, x1, n_samples: 0.5*x0 - 0.7*x1 + np.random.randn(n_samples),
    "x3": lambda x0, x1, n_samples: 0.5*x0 - 0.7*x1 + np.random.randn(n_samples)
}, device="cuda")
adj_dist = adj.AdjMatrixFactorisedDistribution(4)
weight_dist = weight.LinWeightsDistribution(4, weight.MLPs_NodeLinWeightsDistribution)
joint_dist = joint.Factorised_JointDistribution(adj_dist, weight_dist, 4)
data_dist = data.LinearDataDistribution(4)
joint_dist.to("cuda")
train.train1(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(4), data_dist=data_dist,
             init_gibbs_temp=0.5, max_gibbs_temp=50, gibbs_update=True, sparsity_factor=0.1, n_adj_data_samples=256,
             n_weight_data_samples=256, n_adj_joint_dist_samples=1000, n_weight_joint_dist_samples=1, tot_epochs=2000,
             weight_dist_update_steps=10000, adj_dist_update_steps=10, n_interventions=100, predict_intervention=False,
             intervention_scoring_steps=0, adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-3),
             weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))


# lscm = train.LSCM(scm, joint_dist, prior.PriorAdjMatrixDistribution(4), data_dist, 50.)
# optimizer = torch.optim.Adam(weight_dist.parameters())
# adj_matrix = torch.tensor([[[0.,0.,0.,0.],
#                             [0.,0.,0.,0.],
#                             [1.,1.,0.,0.],
#                             [1.,1.,0.,0.]]])
# for e in range(10000):
#     lscm.fit(optimizer, 256, 1, e, include_prior_in_loss=False)
#     print(weight_dist.sample(adj_matrix, return_norm_params=True))


