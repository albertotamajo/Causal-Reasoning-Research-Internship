import torch
from structures import SCM
import numpy as np
from distributions.adjacency import AdjMatrixFactorisedDistribution
from distributions.data import BIC
from distributions.prior import PriorLen2Cycles
from distributions.weight import LinWeightsDistribution, MLPs_NodeLinWeightsDistribution, MLP_NodeLinWeightsDistribution
import torch.nn as nn

torch.manual_seed(0)  #tried seeds: 0-100-22

scm = SCM({
    "x0": lambda     n_samples: np.random.randn(n_samples),
    "x1": lambda x0, n_samples: 0.5*x0 + np.random.randn(n_samples)
}, device="cuda")

adj_samples = torch.tensor([[[0., 0.],
                             [0., 0.]],
                            [[0., 1.],
                             [0., 0.]],
                            [[0., 0.],
                             [1., 0.]],
                            [[0., 1.],
                             [1., 0.]]]).cuda()

real_weights = torch.tensor([[[0., 0.],
                              [0., 0.]],
                             [[0., 0.4],
                              [0., 0.]],
                             [[0., 0.],
                              [0.5, 0.]],
                             [[0., 0.4],
                              [0.5, 0.]]]).cuda()

adj_dist = AdjMatrixFactorisedDistribution(2).cuda()
adj_dist_optimizer = torch.optim.Adam(adj_dist.parameters(), lr=5e-2)
weight_dist = LinWeightsDistribution(2, MLPs_NodeLinWeightsDistribution).cuda()
weight_dist_optimizer = torch.optim.Adam(weight_dist.parameters(), lr=5e-2)
bic = BIC(2)
prior_len2_cycles = PriorLen2Cycles(2)
samples = scm.sample(256)


for _ in range(1000):
    weight_dist_optimizer.zero_grad()
    _, means, _ = weight_dist(adj_samples, return_norm_params=True)
    loss = nn.MSELoss()
    loss = loss(means, real_weights)
    loss.backward()
    weight_dist_optimizer.step()

print(weight_dist(adj_samples))

for _ in range(1000):
    adj_dist_optimizer.zero_grad()
    adj_samples = adj_dist(1)
    _, weight_samples, _ = weight_dist(adj_samples, return_norm_params=True)
    bic_score = bic.log_prob(weight_samples, samples, 1)
    loss = -bic_score - prior_len2_cycles.log_prob(adj_dist.probs(), 150)
    loss.backward()
    adj_dist_optimizer.step()
    print(adj_dist.probs())
