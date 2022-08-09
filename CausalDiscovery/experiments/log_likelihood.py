import torch
from structures import SCM
import numpy as np

scm = SCM({
    "x1": lambda         n_samples: np.random.randn(n_samples),
    "x2": lambda     x1, n_samples: 20 * x1 + np.random.randn(n_samples)
})

mean_param = torch.randn(1, requires_grad=True)
# mean_param size = (1)
adam = torch.optim.RMSprop([mean_param], lr=0.001)
for _ in range(100000):
    adam.zero_grad()
    data_samples = scm.sample(100)
    # data_samples size = (100,2)
    data_samples = torch.transpose(data_samples, dim0=0, dim1=1)
    # data_samples = (2,100)
    weight_samples = torch.distributions.normal.Normal(mean_param, torch.ones(1)).rsample([1000])
    # weight_samples size = (1000,1)
    weight_samples = torch.cat((weight_samples, torch.zeros_like(weight_samples)), dim=1)
    # weight_samples size = (1000,2)
    means = torch.matmul(weight_samples, data_samples)
    # means size = (1000, 100)
    log_prob = torch.distributions.normal.Normal(means, torch.ones_like(means)).log_prob(data_samples[1])
    # log_prob size = (1000, 100)
    log_prob = -log_prob.sum(1)
    # log_prob (1000)
    loss = log_prob.mean()
    loss.backward()
    adam.step()
    print("Mean: ", mean_param)
    #print("Loss: ", loss)
    #print("Average sample: ", weight_samples[0].mean())
