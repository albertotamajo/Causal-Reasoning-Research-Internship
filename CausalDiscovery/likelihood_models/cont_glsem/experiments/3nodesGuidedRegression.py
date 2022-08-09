import torch
from structures import SCM
import numpy as np
from distributions.adjacency import AdjMatrixFactorisedDistribution
from distributions.data import BIC
from distributions.prior import PriorLen2Cycles
from distributions.weight import LinWeightsDistribution, MLPs_NodeLinWeightsDistribution
import itertools
import networkx as nx
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import utils

torch.manual_seed(0)  #TODO: CHANGING SEED HEAVILY AFFECTS TRAINING

scm = SCM({
    "x0": lambda     n_samples: np.random.randn(n_samples),
    "x1": lambda x0, n_samples: 0.5*x0 + np.random.randn(n_samples),
    "x2": lambda x1, n_samples: -0.3*x1 + np.random.randn(n_samples)
}, device="cuda")

samples = scm.sample(100000)

def len3_dags():
    l = [[0.,0.,0.],[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.],[1.,1.,0.],[1.,1.,1.],
         [0.,0.,0.],[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.],[1.,1.,0.],[1.,1.,1.],
         [0.,0.,0.],[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.],[1.,1.,0.],[1.,1.,1.]]
    dags = []
    for m in itertools.combinations(l, 3):
        if m[0][0]==0. and m[1][1]==0. and m[2][2]==0.:
            dags.append(m)
    dags = np.unique(np.asarray(dags), axis=0)
    return dags


def regression_on_dags(dags, samples):
    # dags size = (B,T,T) where B=batch size, T=n_nodes
    # samples size = (L,T) where L=data samples, T=n_nodes
    samples = torch.clone(samples).detach().cpu().numpy()
    weight_matrices = []
    for m in dags:
        coeffs = []
        for i,r in enumerate(m):
            masked_samples = r * samples
            # masked_samples size = (L,T) where L=data samples, T=n_nodes
            coeff = LinearRegression().fit(masked_samples, np.expand_dims(samples[:, i], axis=1)).coef_
            coeffs.append(coeff)
        weight_matrices.append(np.stack(coeffs))
    output = np.stack(weight_matrices)
    return np.squeeze(output, axis=2)


adj_samples = len3_dags()
real_weights = regression_on_dags(adj_samples, samples)
adj_samples = torch.tensor(adj_samples, dtype=torch.float32).cuda()
real_weights = torch.tensor(real_weights, dtype=torch.float32).cuda()

adj_dist = AdjMatrixFactorisedDistribution(3).cuda() #TODO: initialised bern parameters
print(adj_dist.probs())
adj_dist_optimizer = torch.optim.SGD(adj_dist.parameters(), lr=5e-6)  #TODO: SGD seems to have a faster convergence
weight_dist = LinWeightsDistribution(3, MLPs_NodeLinWeightsDistribution).cuda()
weight_dist_optimizer = torch.optim.SGD(weight_dist.parameters(), lr=5e-2)  #TODO: SGD seems to have a faster convergence
bic = BIC(3)
prior_len2_cycles = PriorLen2Cycles(3)
#samples = scm.sample(256)  #TODO: UNCOMMENT HERE

for _ in range(5000):
    weight_dist_optimizer.zero_grad()
    _, means, _ = weight_dist(adj_samples, return_norm_params=True)
    loss = nn.MSELoss()
    loss = loss(means, real_weights)
    loss.backward()
    weight_dist_optimizer.step()

for _ in range(5000):
    adj_dist_optimizer.zero_grad()
    adj_samples_ = adj_dist(1000) #TODO: it was 1 sampling
    _, weight_samples, _ = weight_dist(adj_samples_, return_norm_params=True)
    bic_score = torch.mean(bic.log_prob(weight_samples, samples, 1), dim=0).unsqueeze(1)  #TODO: THIS MEAN SHOULD DISAPPEAR IF SAMPLING IS JUST 1
    regulariser = prior_len2_cycles.log_prob(adj_dist.probs(), 20000) #TODO:CHECK CYCLE TEMPERATURE
    loss = -bic_score - regulariser
    loss.backward()
    adj_dist_optimizer.step()
    print(adj_dist.probs())
    print(loss)
    print(utils.vec_to_adj_mat(adj_dist.params.grad, 3))


def only_dags(adj_samples):
    adj_samples = torch.clone(adj_samples).cpu().detach().numpy()
    dags = []
    for m in adj_samples:
        g = nx.DiGraph(m)
        if nx.is_directed_acyclic_graph(g):
            dags.append(m)
    return torch.tensor(np.stack(dags), dtype=torch.float32).cuda()

adj_samples = only_dags(adj_samples)
_, means, _ = weight_dist(adj_samples, return_norm_params=True)
argmax = torch.argmax(bic.log_prob(means, samples, 1))
print(means[argmax])
