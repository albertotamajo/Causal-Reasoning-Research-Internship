import torch.optim
from distributions import data, joint, prior, weight
import distributions.adjacency as adj
import likelihood_models.train as train
import numpy as np
from structures import SCM

scm = SCM({
    "x0": lambda x1, n_samples: 0.57*x1 + np.random.randn(n_samples),
    "x1": lambda     n_samples: np.random.randn(n_samples),
    "x2": lambda x0, n_samples: -0.3*x0 + np.random.randn(n_samples)
}, device="cuda")

adj_dist = adj.AdjMatrixFactorisedDistribution(3, temp_rsample=0.2)
weight_dist = weight.LinWeightsDistribution(3, weight.MLPs_NodeLinWeightsDistribution)
joint_dist = joint.Factorised_JointDistribution(adj_dist, weight_dist, 3)
data_dist = data.LinearDataDistribution(3)
joint_dist.to("cuda")
#TODO CHECK ARGUMENTS AS THEY MIGHT BE INCORRECT
train.train1(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(3), data_dist=data_dist,
             init_gibbs_temp=0.0, max_gibbs_temp=50, gibbs_update=False, sparsity_factor=0.0, n_adj_data_samples=256,
             n_weight_data_samples=256, n_adj_joint_dist_samples=1000, n_weight_joint_dist_samples=1, tot_epochs=2000,
             weight_dist_update_steps=1000, adj_dist_update_steps=10, n_interventions=1, predict_intervention=False,
             intervention_scoring_steps=0, adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
             weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))

# train.train2(scm=scm, joint_dist=joint_dist, prior_adj_dist=prior.PriorAdjMatrixDistribution(3), data_dist=data_dist,
#              init_gibbs_temp=10., max_gibbs_temp=50, gibbs_update=False, sparsity_factor=0.1, n_adj_data_samples=256,
#              n_weight_data_samples=256, n_adj_joint_dist_samples=1000, n_weight_joint_dist_samples=1, tot_epochs=2000,
#              weight_dist_update_steps=1000, adj_dist_update_steps=20,
#              adj_dist_optimizer=torch.optim.Adam(adj_dist.parameters(), lr=5e-2),
#              weight_dist_optimizer=torch.optim.Adam(weight_dist.parameters(), lr=5e-2))