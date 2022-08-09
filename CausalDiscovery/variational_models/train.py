
def fit(vscm, optimizer, baseline, n_data_samples, n_joint_dist_samples, e):
    """
    Fit adjacency matrix and structural equation parameters variational joint distribution to the true posterior by
    using a Variational Structural Causal Model fitting algorithm
    :param vscm: Variational Structural Causal Model fitting algorithm
    :param optimizer: optimizer
    :param baseline: baseline
    :param n_data_samples: number of samples to be drawn from the real Structural Causal Model
    :param n_joint_dist_samples: number of samples to be drawn from the adjacency matrix and structural equation
                                 parameters variational joint distribution
    :param e: epoch
    :return: elbo loss, negative log likelihood of data, Kullback-leibler divergence between variational joint
             distribution and prior distribution, baseline
    """
    optimizer.zero_grad()
    data_likelihood, kl, var_log_probs = vscm(n_data_samples, n_joint_dist_samples, e)  # TODO: Check if additional entropy regularization is required
    # data_likelihood size = (L,1) where L=n_joint_dist_samples
    # kl size = (L,1) where L=n_joint_dist_samples
    # var_log_probs size = (L,1) where L=n_joint_dist_samples
    score_val = (- data_likelihood + kl).detach()  # TODO: check what it is doing here. It seems incorrect
    # score val size = (L,1) where L=n_joint_dist_samples
    per_sample_elbo = var_log_probs * (score_val - baseline)
    # per_sample_elbo size = (L,1) where L=n_joint_dist_samples
    baseline = 0.95 * baseline + 0.05 * score_val.mean()
    loss = per_sample_elbo.mean()
    # loss size = (1)
    loss.backward()
    optimizer.step()

    data_likelihood_ = -data_likelihood.mean().item()
    # data_likelihood_ size = (1)
    kl_ = kl.mean().item()
    # kl_ size = (1)
    loss_ = (-data_likelihood + kl).mean().item()
    # loss size = (1)
    return loss_, data_likelihood_, kl_, baseline

    # optimizer.zero_grad()
    # data_likelihood, kl, var_log_probs = vscm(n_data_samples, n_joint_dist_samples, e)  # TODO: Check if additional entropy regularization is required
    # # data_likelihood size = (L,1) where L=n_joint_dist_samples
    # # kl size = (L,1) where L=n_joint_dist_samples
    # # var_log_probs size = (L,1) where L=n_joint_dist_samples
    # loss = (-data_likelihood + kl).mean()
    # # loss size = (1)
    # loss.backward()
    # optimizer.step()
    #
    # data_likelihood_ = -data_likelihood.mean().item()
    # # data_likelihood_ size = (1)
    # kl_ = kl.mean().item()
    # # kl_ size = (1)
    # loss_ = (-data_likelihood + kl).mean().item()
    # # loss size = (1)
    # return loss_, data_likelihood_, kl_, baseline

