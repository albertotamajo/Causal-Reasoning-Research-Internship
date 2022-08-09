import torch
import torch.nn.functional as F
from structures import SCM


def sample_adjacency_matrix(parameters: torch.tensor):
    """
    Sample adjacency matrix given the bernoulli success parameter provided for each entry.
    :param parameters: matrix of bernoulli success parameters for each entry
    :return: adjacency matrix
    """
    # parameters size = (T,T) where T=endogenous variables
    parameters = torch.flatten(parameters).unsqueeze(1)
    # parameters size = (T*T, 1) where T=endogenous variables
    fail_parameters = torch.ones_like(parameters) - parameters
    # fail_parameters size = (T*T, 1) where T=endogenous variables
    logits = torch.log(torch.cat((parameters, fail_parameters), dim=1))
    # logits size = (T*T, 2) where T=endogenous variables
    one_hots = F.gumbel_softmax(logits, hard=True)
    # one_hots size = (T*T, 2) where T=endogenous variables
    adjacency_matrix = torch.matmul(one_hots, torch.tensor([1., 0.]))  # TODO should the tensor be autograded?
    # adjaency_matrix size = (T*T) where T=endogenous variables
    adjacency_matrix = adjacency_matrix.resize_as(parameters)
    # adjacency_matrix size = (T,T) where T=endogenous variables
    return adjacency_matrix


def compute_target_means(input: torch.tensor, O: torch.tensor):
    """
    Compute target means given a batch of structural equation parameters and a batch of data samples.
    The underlying assumption is that the structural equations are linear.
    :param input: batch of data samples
    :param O: batch of structural equation parameters
    :return: target means
    """
    # O size = (L,T,T+1) where L= variational joint distribution samples, T=endogenous variables
    # input size = (B,T) where B=data samples, T=endogenous variables
    # Augment input with a column of 1s at index 0
    B, _ = input.shape
    input = torch.cat((torch.ones((B, 1)), input), dim=1)
    # input size = (B, T+1) where B=data samples, T=endogenous variables
    input = torch.transpose(input, 0, 1)
    # input size = (T+1, B) where B=data samples, T=endogenous variables
    target = torch.matmul(O, input)
    # target size = (L, T, B) where L= variational joint distribution samples, B=data samples, T=endogenous variables
    target = torch.transpose(target, 1, 2)
    # target size = (L, B, T) where L= variational joint distribution samples, B=data samples, T=endogenous variables
    return target


# def gaussian_likelihood(input: torch.tensor, target: torch.tensor):
#     """
#     Compute the gaussian likelihood of a batch of samples drawn from the underlying Structural Causal Model given a
#     batch of means sampled from the variational joint distribution
#     :param input: batch of samples drawn from the underlying Structural Causal Model
#     :param target: batch of means sampled from the variational joint distribution
#     :return: gaussian likelihood
#     """
#     # input size = (B,T) where B=data samples, T=endogenous variables
#     # target size = (L,B,T) where L= variational joint distribution samples, B=data samples, T=endogenous means
#     likelihood_loss = torch.nn.GaussianNLLLoss(reduction='none')
#     output_likelihood = torch.sum(likelihood_loss(input, target, torch.ones(input.shape[0]).unsqueeze(-1)), dim=(2, 1))
#     # output_likelihood size = (L) where L= variational joint distribution samples
#     return output_likelihood


def elbo_loss(input: torch.tensor, target: torch.tensor, var_likelihood: torch.tensor, prior_likelihood: torch.tensor):
    """
    Compute elbo loss given a batch of samples drawn from the underlying Structural Causal Model and a batch of
    means sampled from the variational joint distribution
    :param input: batch of samples drawn from the underlying Structural Causal Model
    :param target: batch of means sampled from the variational joint distribution
    :param var_likelihood: likelihoods of the graphs and structural equations drawn from the variational joint
                           distribution with respect to the variational joint distribution
    :param prior_likelihood: likelihoods of the graphs and structural equations drawn from the variational joint
                             distribution with respect to the prior joint distribution
    :return: elbo loss
    """
    # input size = (B,T) where B=data samples, T=endogenous variables
    # target size = (L,B,T) where L= variational joint distribution samples, B=data samples, T=endogenous means
    # var_likelihood size = (L) where L= variational joint distribution samples
    # prior_likelihood = (L) where L= variational joint distribution samples
    likelihood_loss = torch.nn.GaussianNLLLoss(reduction='none')
    output_likelihood = torch.sum(likelihood_loss(input, target, torch.ones(input.shape[0]).unsqueeze(-1)), dim=(2, 1))
    # output_likelihood size = (L) where L= variational joint distribution samples
    output_kl_divergence = torch.log(var_likelihood) - torch.log(prior_likelihood)
    # output_kl_divergence size = (L) where L= variational joint distribution samples
    return torch.mean(output_likelihood + output_kl_divergence)


def fit_observational_data(M:SCM, Q, P, B, L, optimizer):
    """
    Fit joint distribution of the graphs and structural parameters to observational data drawn from an underlying
    Structural Causal Model.
    :param M: underlying Structural Causal Model
    :param Q: variational joint distribution of the graphs and structural equation parameters
    :param P: prior joint distribution of the graphs and structural equation parameters
    :param B: batch size
    :param L: number of samples drawn from Q
    :param optimizer: optimizer
    :return: None
    """
    # Clear gradients
    optimizer.zero_grad()
    D = M.sample(B)
    # D size = (B,T) where B=data samples, T=endogenous variables
    (G, O) = Q.sample(L)
    # G size = (L, T, T) where L= variational joint distribution samples, T=endogenous means
    # O size = (L, T, T+1) where L= variational joint distribution samples, T=endogenous variables
    # Compute elbo loss
    loss = elbo_loss(D, compute_target_means(D, O), var_likelihood, prior_likelihood)
    #Optimise
    loss.backward()
    optimizer.step()


def predict_intervention(M_:SCM, Q, W, B, L):
    """
    Predict intervened variable.
    :param M_: underlying intervened Structural Causal Model
    :param Q: variational joint distribution of the graphs and structural equation parameters
    :param W: scoring iterations
    :param B: batch size
    :param L: number of samples drawn from Q
    :return: index of intervened variable
    """
    total_scores = torch.zeros(len(M_))
    # total_scores size = (T) where T=endogenous variables
    for _ in range(W):
        D = M_.sample(B)
        # D size = (B,T) where B=data samples, T=endogenous variables
        (G, O) = Q.sample(L)
        # G size = (L, T, T) where L= variational joint distribution samples, T=endogenous variables
        # O size = (L, T, T+1) where L= variational joint distribution samples, T=endogenous variables
        target = compute_target_means(D, O)
        # target size = (L,B,T) where L=variational joint distribution samples, B=data samples, T=endogenous variables
        likelihood_loss = torch.nn.GaussianNLLLoss(reduction='none')
        output_likelihood = torch.sum(likelihood_loss(D, target, torch.ones(D.shape[0]).unsqueeze(-1)), dim=1)
        # output_likelihood size = (L,T) where L= variational joint distribution samples, T=endogenous variables
        output_likelihood = torch.mean(output_likelihood, dim=0)
        # output_likelihood size = (T) where T=endogenous variables
        total_scores = total_scores + output_likelihood
        # total_scores size = (T) where T=endogenous variables
    return torch.argmax(total_scores)


def fit_interventional_data(M_:SCM, Q, P, B, L, n, optimizer):
    """
    Fit joint distribution of the graphs and structural parameters to interventional data drawn from an underlying
    intervened Structural Causal Model.
    :param M_: underlying intervened Structural Causal Model
    :param Q: variational joint distribution of the graphs and structural equation parameters
    :param P: prior joint distribution of the graphs and structural equation parameters
    :param B: batch size
    :param L: number of samples drawn from Q
    :param n: index of intervened variable
    :param optimizer: optimizer
    :return: None
    """
    # Clear gradients
    optimizer.zero_grad()
    D = M_.sample(B)
    # D size = (B,T) where B=data samples, T=endogenous variables
    (G, O) = Q.sample(L)
    # G size = (L, T, T) where L= variational joint distribution samples, T=endogenous means
    # O size = (L, T, T+1) where L= variational joint distribution samples, T=endogenous variables
    # Remove intervened variable from data samples
    D_ = torch.cat((D[:, 0:n], D[:, n+1:]), dim=1)
    # D_ size = (B,T-1) where B=data samples, T=endogenous variables
    # Remove intervened variable's structural equation parameters
    O_ = torch.cat((O[:, 0:n, :], O[:, n+1:, :]), dim=1)
    # O_ size = (L, T-1, T+1) where L= variational joint distribution samples, T=endogenous variables
    # Compute elbo loss
    loss = elbo_loss(D_, compute_target_means(D_, O_), var_likelihood, prior_likelihood)
    # Optimise
    loss.backward()
    optimizer.step()


def run_SCM_variational_inference(M:SCM, Q, P, I, F, R, W, Z, B, L, optimizer, predict_intervened_variable=True):
    """
    Run the Variational SCM Inference Algorithm.
    :param M: underlying Structural Causal Model
    :param Q: variational joint distribution of the graphs and structural equation parameters
    :param P: prior joint distribution of the graphs and structural equation parameters
    :param I: epoch iterations
    :param F: observational fitting training iterations
    :param R: intervention iterations
    :param W: scoring iterations
    :param Z: interventional fitting training iterations
    :param B: batch size
    :param L: number of samples drawn from Q
    :param optimizer: optimizer
    :param predict_intervened_variable: whether to predict index of intervened node or not
    :return: None
    """
    for i in range(I):
        for _ in range(F):
            fit_observational_data(M, Q, P, B, L, optimizer)
        P = Q # This should be a deep copy TODO
        for _ in range(R):
            n = torch.randint(0, len(M))
            M_ = M.do(n)  # Intervene on variable at index n
            if predict_intervened_variable:
                n = predict_intervention(M_, Q, W, B, L)
            for _ in range(Z):
                fit_interventional_data(M_, Q, P, B, L, n, optimizer)
        P = Q # This should be a deep copy TODO






