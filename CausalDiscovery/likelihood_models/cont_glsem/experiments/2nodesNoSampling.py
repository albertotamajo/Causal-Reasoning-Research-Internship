import torch

import utils
from structures import SCM
import numpy as np
from distributions.adjacency import AdjMatrixFactorisedDistribution
from distributions.data import BIC
from distributions.prior import PriorLen2Cycles

torch.manual_seed(0)

scm = SCM({
    "x0": lambda  x1,   n_samples: 0.5*x1 + np.random.randn(n_samples),
    "x1": lambda n_samples: np.random.randn(n_samples)
}, device="cuda")

weight_dist = torch.tensor([[0.5], [0.5]], requires_grad=True, device="cuda")
adj_dist_optimizer = torch.optim.Adam([weight_dist], lr=0.1)
bic = BIC(2)
prior_len2_cycles = PriorLen2Cycles(2)
samples = scm.sample(2000)

for i in range(1000):
    adj_dist_optimizer.zero_grad()
    weights = utils.vec_to_adj_mat(weight_dist, 2)
    bic_score = bic.log_prob(weights, samples, 1)
    loss = -bic_score - prior_len2_cycles.log_prob(weights, 200)
    loss.backward()
    adj_dist_optimizer.step()
    print(utils.vec_to_adj_mat(weight_dist, 2))