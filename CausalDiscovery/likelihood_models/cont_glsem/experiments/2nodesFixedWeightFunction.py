import torch
from structures import SCM
import numpy as np
from distributions.adjacency import AdjMatrixFactorisedDistribution
from distributions.data import BIC
from distributions.prior import PriorLen2Cycles

torch.manual_seed(0)

scm = SCM({
    "x0": lambda x1, n_samples: 0.5*x1 + np.random.randn(n_samples),
    "x1": lambda     n_samples: np.random.randn(n_samples)
}, device="cuda")

adj_dist = AdjMatrixFactorisedDistribution(2).cuda()
adj_dist_optimizer = torch.optim.Adam(adj_dist.parameters(), lr=5e-2)
bic = BIC(2)
prior_len2_cycles = PriorLen2Cycles(2)
samples = scm.sample(256)

for _ in range(1000):
    adj_dist_optimizer.zero_grad()
    adj_samples = adj_dist(1000)
    weight_samples = adj_samples * torch.tensor([[[0., 0.5], [0.4, 0.]]]).to(device=adj_samples.device)
    bic_score = torch.mean(bic.log_prob(weight_samples, samples, 1), dim=0)
    loss = -bic_score - prior_len2_cycles.log_prob(adj_dist.probs(), 150)
    loss.backward()
    adj_dist_optimizer.step()
    print(adj_dist.probs())
