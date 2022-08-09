import torch.optim
import distributions.adjacency as adj
import distributions.weight as weight
from distributions.data import LinearDataDistribution, BIC
from distributions.prior import PriorLen2Cycles
import utils
from structures import SCM
import numpy as np


class CONT_GLSEM:
    def __init__(self, n_nodes, adj_dist, weight_dist, adj_dist_optimizer, weight_dist_optimizer):
        assert n_nodes == adj_dist.n_nodes and n_nodes == weight_dist.n_nodes
        self.n_nodes = n_nodes
        self.adj_dist = adj_dist
        self.weight_dist = weight_dist
        self.adj_dist_optimizer = adj_dist_optimizer
        self.weight_dist_optimizer = weight_dist_optimizer
        self.likelihood = LinearDataDistribution(self.n_nodes)
        self.bic = BIC(self.n_nodes)
        self.prior2cycle = PriorLen2Cycles(self.n_nodes)

    def fit(self, data, epochs, adj_update_steps=10, regression_steps=10000, regression_adj_samples=1000,
            one_time_regression=False, var=1.0, lam=150):
        for e in range(epochs):
            if (not one_time_regression) or (one_time_regression and e==0):
                for _ in range(regression_steps):
                    self.weight_dist_optimizer.zero_grad()
                    adj_sample = self.adj_dist(regression_adj_samples)
                    _, means, _ = self.weight_dist(adj_sample, return_norm_params=True)
                    loss = - ((self.likelihood.log_prob(means, data)).mean())
                    loss.backward()
                    self.weight_dist_optimizer.step()
            for _ in range(adj_update_steps):
                adj_sample = self.adj_dist(1)
                means = adj_sample * torch.tensor([[[0., 0.5], [0.4, 0.]]]).to(device=adj_sample.device)
                #_, means, _ = self.weight_dist(adj_sample, return_norm_params=True) #todo: fix here
                bic_score = self.bic.log_prob(means, data, var)
                probs = utils.vec_to_adj_mat(self.adj_dist._to_probability().unsqueeze(0), self.n_nodes)
                loss = -bic_score - self.prior2cycle.log_prob(probs, lam)
                loss.backward()
                self.adj_dist_optimizer.step()

            print(self.adj_dist.sample(1, return_probs=True)[1])  # TODO DELETE THESE PRINT LINES

def cycle_constraint(adj_dist):  #TODO: DELETE THIS FUNCTION AFTER REFORMATTING CODE FOR THE BIC VARIANT ALGORITHM
    probs = adj_dist._to_probability().unsqueeze(0)
    # probs size = (1,T,1) where T=self.n_dim_out
    probs = utils.vec_to_adj_mat(probs, 2).squeeze(0)
    probs_transpose = probs.T
    return torch.cosh(probs * probs_transpose).sum()


scm = SCM({
    "x0": lambda x1, n_samples: 0.5 * x1 + np.random.randn(n_samples),
    "x1": lambda     n_samples: np.random.randn(n_samples)
}, device="cuda")
adj_dist = adj.AdjMatrixFactorisedDistribution(2).cuda()
weight_dist = weight.LinWeightsDistribution(2, weight.MLP_NodeLinWeightsDistribution).cuda()
adj_dist_optimizer = torch.optim.Adam(adj_dist.parameters())
weight_dist_optimizer = torch.optim.Adam(weight_dist.parameters())

cont_glsem = CONT_GLSEM(2, adj_dist, weight_dist, adj_dist_optimizer, weight_dist_optimizer)
cont_glsem.fit(scm.sample(2000), 10000, regression_steps=0, regression_adj_samples=10, one_time_regression=True)


