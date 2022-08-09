from distributions import weight, data, prior, joint
import distributions.adjacency as adj
from structures import SCM
import torch
import numpy as np
import utils


real_scm = SCM({
    "x1": lambda         n_samples: np.random.randn(n_samples),
    "x2": lambda         n_samples: np.random.randn(n_samples),
    "x3": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples),
    "x4": lambda x1, x2, n_samples: 0.5*x1 - 0.7*x2 + np.random.randn(n_samples)
})

fake_scm = SCM({
    "x1": lambda         n_samples: np.random.randn(n_samples),
    "x2": lambda     x3, n_samples: -0.2232*x3 + np.random.randn(n_samples),
    "x3": lambda x1, x2, n_samples: 0.5020*x1 - 0.6897*x2 + np.random.randn(n_samples),
    "x4": lambda x1, x2, n_samples: 0.5027*x1 - 0.7*x2 + np.random.randn(n_samples)
})

real_adj_matrix = torch.tensor([[[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]])
fake_adj_matrix = torch.tensor([[[0.,0.,0.,0.], [0.,0.,1.,0.], [1.,1.,0.,0.], [1.,1.,0.,0.]]])
real_weights = torch.tensor([[[0., 0., 0., 0.], [0., 0., 0., 0.], [0.5, -0.7, 0.0, 0.0], [0.5, -0.7, 0.0, 0.0]]])
fake_weights = torch.tensor([[[0., 0., 0., 0.], [0., 0., -0.2232, 0.], [0.5020, -0.6897, 0.0, 0.0], [0.5027, -0.7, 0.0, 0.0]]])
likelihood = data.LinearDataDistribution(4)
real_likelihoods = []
fake_likelihoods = []
for i in range(10000):
    data_samples = real_scm.sample(1000)
    real_likelihood = -likelihood.log_prob(real_weights, data_samples).item()
    fake_likelihood = -likelihood.log_prob(fake_weights, data_samples).item()
    real_likelihoods.append(real_likelihood)
    fake_likelihoods.append(fake_likelihood)

print("Correct adjacency matrix likelihood: ", torch.tensor(real_likelihoods).mean())
print("Incorrect adjacency matrix likelihood: ", torch.tensor(fake_likelihoods).mean())

