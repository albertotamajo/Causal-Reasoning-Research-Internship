import torch
import utils
import torch.nn as nn
import torch.nn.functional as F


class AdjMatrixFactorisedDistribution(nn.Module):
    """
    Adjacency matrix factorised distribution
    """
    def __init__(self, n_nodes, temp_rsample=1.0, init_params=None):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        :param temp_rsample: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param init_params: if None, the parameters are initialised so that the probability of there being an edge
                            in the off-diagonal entries of the generated adjacency matrices is 0.5. Otherwise, a tensor
                            of shape (n_nodes * (n_nodes - 1), 1) should be provided to initialise the parameters.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim_out = n_nodes * (n_nodes - 1)
        self.temp_rsample = temp_rsample
        self.register_buffer('one_hot_encoder', torch.tensor([1., 0.]))
        if init_params is None:
            self.params = torch.nn.Parameter(torch.zeros(self.n_dim_out, 1))
            # params size = (T,1) where T=self.n_dim_out
        else:
            assert init_params.shape == (self.n_dim_out, 1)
            self.params = torch.nn.Parameter(init_params)
            # params size = (T,1) where T=self.n_dim_out

    def forward(self, batch_size, return_probs=False, reparametrized=True, temp=None):
        """
        Sample a batch of adjacency matrices from the factorised distribution
        :param batch_size: number of adjacency matrices to be sampled
        :param return_probs: if True, the Bernoulli success probabilities are returned as well
        :param reparametrized: if False, every element of the adjacency matrices is sampled using a Bernoulli
                               distribution. Sampling from the Bernoulli distribution is not differentiable, though. If
                               True, every element of the adjacency matrices is sampled using the Gumbel-Softmax which
                               approximates a Bernoulli distribution through a continuous function. Thus, the
                               Gumbel-Softmax is differentiable.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :return: adjacency matrix samples, Bernoulli success probabilities
        """
        samples, probs = self._sample(batch_size, reparametrized=reparametrized, temp=temp)
        # samples size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        if return_probs:
            return samples, probs
        else:
            return samples

    def sample(self, batch_size, return_probs=False):
        """
        Sample a batch of adjacency matrices from the factorised distribution using the Bernoulli distribution.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_probs: if True, the Bernoulli success probabilities are returned as well
        :return: adjacency matrix samples or adjacency matrix samples, Bernoulli success probabilities
        """
        samples, probs = self._sample(batch_size, reparametrized=False)
        # samples size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        if return_probs:
            return samples, probs
        else:
            return samples

    def rsample(self, batch_size, return_probs=False, temp=None):
        """
        Sample a batch of adjacency matrices from the factorised distribution using the Gumbel-Softmax Bernoulli
        approximation.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_probs: if True, the Bernoulli success probabilities are returned as well
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :return: adjacency matrix samples or adjacency matrix samples, Bernoulli success probabilities
        """
        samples, probs = self._sample(batch_size, reparametrized=True, temp=temp)
        # samples size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        if return_probs:
            return samples, probs
        else:
            return samples

    def _sample(self, batch_size, reparametrized=False, temp=None):
        """
        Sample a batch of adjacency matrices from the factorised distribution
        :param batch_size: number of adjacency matrices to be sampled
        :param reparametrized: if False, every element of the adjacency matrices is sampled using a Bernoulli
                               distribution. Sampling from the Bernoulli distribution is not differentiable, though. If
                               True, every element of the adjacency matrices is sampled using the Gumbel-Softmax which
                               approximates a Bernoulli distribution through a continuous function. Thus, the
                               Gumbel-Softmax is differentiable.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :return: adjacency matrix samples, Bernoulli success probabilities
        """
        if temp is None:
            temp = self.temp_rsample

        probs = self._to_probability()
        # probs size = (T,1) where T=self.n_dim_out
        probs = probs.unsqueeze(0).expand(batch_size, self.n_dim_out, 1)
        # probs size = (B,T,1) where B=batch size, T=self.n_dim_out

        if reparametrized:
            fail_probs = torch.ones_like(probs) - probs
            # fail_probs size = (B,T,1) where T=self.n_dim_out
            logits = torch.log(torch.cat((probs, fail_probs), dim=2))
            # logits size = (B,T,2) where B=batch size, T=self.n_dim_out
            one_hots = F.gumbel_softmax(logits, hard=True, dim=2, tau=temp)
            # one_hots size = (B,T,2) where B=batch size, T=self.n_dim_out
            _sample = torch.matmul(one_hots, self.one_hot_encoder).unsqueeze(2)
            # _sample size = (B,T,1) where T=self.n_dim_out
        else:
            _sample = torch.distributions.Bernoulli(probs=probs).sample()
            # _sample size = (B,T,1) where B=batch size, T=self.n_dim_out
        samples = utils.vec_to_adj_mat(_sample, self.n_nodes)
        # samples size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        probs = utils.vec_to_adj_mat(probs, self.n_nodes)
        # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        return samples, probs

    def log_prob(self, adj_matrices, return_probs=False):
        """
        Compute the log probability of the given batch of adjacency matrices.
        :param adj_matrices: batch of adjacency matrices
        :param return_probs: if True, the Bernoulli success probabilities for each entry of the adjacency matrices is
                             returned as well
        :return: log probability for the batch of adjacency matrices or log probability for the batch of adjacency
                 matrices and the Bernoulli success probabilities for each entry of the adjacency matrices
        """
        # adj_matrices size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        assert adj_matrices.shape[1] == self.n_nodes and adj_matrices.shape[2] == self.n_nodes
        probs = self._to_probability()
        # probs size = (T,1) where T=self.n_dim_out
        probs = probs.unsqueeze(0).expand(adj_matrices.shape[0], self.n_dim_out, 1)
        # probs size = (B,T,1) where B=batch size, T=self.n_dim_out
        probs = utils.vec_to_adj_mat(probs, self.n_nodes)
        # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        log_probs = torch.distributions.Bernoulli(probs=probs).log_prob(adj_matrices)
        # log_probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        log_probs = torch.sum(log_probs, dim=(2, 1)).unsqueeze(1)
        # log_probs size = (B,1)
        if return_probs:
            return log_probs, probs
        return log_probs

        # value = utils.adj_mat_to_vec(adj_matrices, self.n_nodes).unsqueeze(2)
        # # value size = (B,T,1) where B=batch size, T=self.n_dim_out
        # probs = self._to_probability()
        # # probs size = (T,1) where T=self.n_dim_out
        # log_probs = torch.distributions.Bernoulli(probs=probs).log_prob(value).sum(1)
        # # log_probs size = (B,1)
        # if return_probs:
        #     probs = probs.unsqueeze(0).expand(adj_matrices.shape[0], self.n_dim_out, 1)
        #     # probs size = (B,T,1) where B=batch size, T=self.n_dim_out
        #     probs = utils.vec_to_adj_mat(probs, self.n_nodes)
        #     # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        #     return log_probs, probs
        # return log_probs  TODO: old block of code. Check the new one well before deleting this

    def entropy(self, n_samples=10 ** 6):  # TODO: to be checked
        """

        :param n_samples:
        :return:
        """
        bs = 100000
        curr = 0
        ent = 0.
        while curr < n_samples:
            curr_batch_size = min(bs, n_samples - curr)
            ent -= torch.sum(self.log_prob(self.sample([curr_batch_size])))
            curr += curr_batch_size
        return ent / n_samples

    def mode(self, return_logprob=False):
        """
        Compute the mode of the adjacency matrix distribution
        :param return_logprob: if True, the log probability associated with the mode is returned as well
        :return: mode of the adjacency matrix distribution or mode of the adjacency matrix distribution and
                 its log probability
        """
        probs = self._to_probability().unsqueeze(0)
        # probs size = (1,T,1) where T=self.n_dim_out
        probs = utils.vec_to_adj_mat(probs, self.n_nodes)
        # probs size = (1,Q,Q) where Q=self.n_nodes
        mode_ = torch.round(probs)
        # mode_ size = (1,Q,Q) where Q=self.n_nodes
        if return_logprob:
            log_prob = self.log_prob(mode_)
            # log_prob size = (1,1)
            return mode_, log_prob
        return mode_

    def _to_probability(self):
        """
        Convert the parameters into probabilities
        :return: vector of probabilities
        """
        return 1/(1+torch.exp(- self.params / 2))  #TODO: this modified version of the sigmoid seems to work better
                                                   # than the classic one

    def probs(self):
        """
        Return the Bernoulli parameters describing the factorised adjacency matrix distribution
        :return: Bernoulli parameters describing the factorised adjacency matrix distribution
        """
        probs = self._to_probability().unsqueeze(0)
        # probs size = (1,T,1) where T=self.n_dim_out
        probs = utils.vec_to_adj_mat(probs, self.n_nodes)
        # probs size = (1,Q,Q) where Q=self.n_nodes
        return probs

    def zero_grad_except_parent(self, n):
        """
        Zero out gradient for all of the Bernoulli parameters except for those that denote the probability
        of a node to have a given node as parent.
        :param n: index of parent node
        :return: hook
        """
        zeros = torch.zeros_like(self.params).unsqueeze(0)
        # zeros size = (1,T,1) where T=self.n_dim_out
        zeros = utils.vec_to_adj_mat(zeros, self.n_nodes)
        # zeros size = (1,Q,Q) where Q=self.n_nodes
        zeros[:,:,n]= 1.
        zeros = utils.adj_mat_to_vec(zeros, self.n_nodes)
        # zeros size = (1,T) where T=self.n_dim_out
        zeros = torch.transpose(zeros, dim0=0, dim1=1).contiguous()
        # zeros size = (T,1) where T=self.n_dim_out
        hook = self.params.register_hook(lambda grad: grad * zeros)
        return hook



class LSTM_AdjMatrixDistribution(nn.Module):
    """
    Adjacency matrix autoregressive distribution using an LSTM
    """
    def __init__(self, n_nodes, hidden_dim=48, n_layers=3, temp_rsample=1.0):
        """
        Initialise distribution
        :param n_nodes: number of nodes in the graph
        :param hidden_dim: LSTM input, hidden and cell states dimension
        :param n_layers: number of layers in the LSTM
        :param temp_rsample: non-negative scalar temperature for the Gumbel-Softmax distribution
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim_out = n_nodes * (n_nodes - 1)  # number of elements off the adjacency matrix diagonal
        self.hidden_dim = hidden_dim
        self.n_classes = 1
        self.n_layers = n_layers
        self.temp_rsample = temp_rsample

        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=n_layers, batch_first=True)
        # Projects the output of the LSTM into a bernoulli success probability
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.n_classes), nn.Sigmoid())
        # Embeds the input of the LSTM into a vector
        self.embed = nn.Linear(self.n_classes, self.hidden_dim)
        # Initialise the initial hidden state. It will be updated when using back-propagation
        self.h0 = nn.Parameter(1e-3 * torch.randn(1, self.n_layers, self.hidden_dim))
        # Initialise the initial cell state. It will be updated when using back-propagation
        self.c0 = nn.Parameter(1e-3 * torch.randn(1, self.n_layers, self.hidden_dim))
        # Initialise variable for the initial input of the LSTM. It will be updated when using back-propagation
        self._init_input_param = nn.Parameter(torch.zeros(1, 1, self.n_classes))
        self.register_buffer('one_hot_encoder', torch.tensor([1., 0.]))

    def forward(self, batch_size, reparametrized=True, temp=None, return_states=False, start_state=None,
                init_input=None):
        """
        Sample a batch of adjacency matrices from the autoregressive distribution
        :param batch_size: number of adjacency matrices to be sampled
        :param reparametrized: if False, every element of the adjacency matrices is sampled using a Bernoulli
                               distribution. Sampling from the Bernoulli distribution is not differentiable, though. If
                               True, every element of the adjacency matrices is sampled using the Gumbel-Softmax which
                               approximates a Bernoulli distribution through a continuous function. Thus, the
                               Gumbel-Softmax is differentiable.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param return_states: if True also LSTM states and Bernoulli success probabilities are returned
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples or adjacency matrix samples, LSTM states, Bernoulli success probabilities
        """
        samples, states, probs = self._sample(batch_size, reparametrized=reparametrized, temp=temp,
                                              start_state=start_state, init_input=init_input)
        if return_states:
            return samples, states, probs
        else:
            return samples

    def _compute_outputs_states(self, inputs, state):
        """
        Compute batch of outputs and states by feeding a batch of inputs and initial states to the LSTM
        :param inputs: batch of inputs to the LSTM
        :param state: batch of initial states of the LSTM
        :return: batch of scalar probabilities, batch of output states
        """
        # input size = (B,L,1) where B=batch size, L=sequence length, 1=self.n_classes
        # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
        inputs = self.embed(inputs)
        # inputs size = (B,L,H) where B=batch size, L=sequence length, H=self.hidden_dim
        out, state = self.rnn(inputs, self._t(state))
        # out size = (B,L,H) where B=batch size, L=sequence length, H=self.hidden_dim
        # state = tuple where both elements have size (Q,B,H) where Q=self.n_layers, B=batch size, H=self.hidden_dim
        state = self._t(state)
        # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
        prob = self.proj(out)
        # prob size = (B,L,1) where B=batch size, L=sequence length, 1=self.n_classes
        return prob, state

    def sample(self, batch_size, return_states=False, start_state=None, init_input=None):
        """
        Sample a batch of adjacency matrices from the autoregressive distribution using the Bernoulli distribution.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_states: if True also LSTM states and Bernoulli success probabilities are returned
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples or adjacency matrix samples, LSTM states, Bernoulli success probabilities
        """
        samples, states, probs = self._sample(batch_size, reparametrized=False, start_state=start_state,
                                              init_input=init_input)
        if return_states:
            return samples, states, probs
        else:
            return samples

    def rsample(self, batch_size, return_states=False, temp=None, start_state=None, init_input=None):
        """
        Sample a batch of adjacency matrices from the autoregressive distribution using the Gumbel-Softmax Bernoulli
        approximation.
        :param batch_size: number of adjacency matrices to be sampled
        :param return_states: if True also LSTM states and Bernoulli success probabilities are returned
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples or adjacency matrix samples, LSTM states, Bernoulli success probabilities
        """
        samples, states, probs = self._sample(batch_size, reparametrized=True, temp=temp, start_state=start_state,
                                              init_input=init_input)
        if return_states:
            return samples, states, probs
        else:
            return samples

    def _sample(self, batch_size, reparametrized=False, temp=None, start_state=None, init_input=None):
        """
        Sample a batch of adjacency matrices from the autoregressive distribution
        :param batch_size: number of adjacency matrices to be sampled
        :param reparametrized: if False, every element of the adjacency matrices is sampled using a Bernoulli
                               distribution. Sampling from the Bernoulli distribution is not differentiable, though. If
                               True, every element of the adjacency matrices is sampled using the Gumbel-Softmax which
                               approximates a Bernoulli distribution through a continuous function. Thus, the
                               Gumbel-Softmax is differentiable.
        :param temp: non-negative scalar temperature for the Gumbel-Softmax distribution
        :param start_state: initial state of the LSTM from which starting sampling
        :param init_input: initial input of the LSTM from which starting sampling
        :return: adjacency matrix samples, LSTM states, Bernoulli success probabilities
        """
        if temp is None:
            temp = self.temp_rsample
        if start_state is None:
            state = self._get_state(batch_size)  # hidden / cell state at t=0
        else:
            state = start_state
        if init_input is None:
            input = self._get_init_input(batch_size)  # input at t=0
        else:
            input = init_input

        # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
        # input size = (B,1,1) where B=batch size

        sampled_tokens = []
        state_array_1 = []
        state_array_2 = []
        prob_array = []

        for t in range(self.n_dim_out):
            probs, state = self._compute_outputs_states(input, state)
            # prob size = (B,1,1) where B=batch size
            # state = tuple where both elements have size (B,Q,H) where B=batch size, Q=self.n_layers, H=self.hidden_dim
            if reparametrized:
                fail_probs = torch.ones_like(probs) - probs
                # fail_probs size = (B,1,1) where B=batch size
                logits = torch.log(torch.cat((probs, fail_probs), dim=2))
                # logits size = (B,1,2) where B=batch size
                one_hots = F.gumbel_softmax(logits, hard=True, dim=2, tau=temp)
                # one_hots size = (B,1,2) where B=batch size
                _sample = torch.matmul(one_hots, self.one_hot_encoder).unsqueeze(2)
                # _sample size = (B,1,1) where B=batch size
            else:
                _sample = torch.distributions.Bernoulli(probs=probs).sample()
                # _sample size = (B,1,1) where B=batch size
            input = _sample
            sampled_tokens.append(_sample)
            state_array_1.append(state[0])
            state_array_2.append(state[1])
            prob_array.append(probs)

        samples = torch.cat(sampled_tokens, dim=1)
        # samples size = (B,T,1) where B=batch size, T=self.n_dim_out
        states = [torch.stack(state_array_1, dim=1), torch.stack(state_array_2, dim=1)]
        # states = tuple where both elements have size (B,T,Q,H) where B=batch size, T=self.n_dim_out, Q=self.n_layers,
        # H=self.hidden_dim
        probs = torch.cat(prob_array, dim=1)
        # probs size = (B,T,1) where B=batch size, T=self.n_dim_out

        samples = utils.vec_to_adj_mat(samples, self.n_nodes)
        # samples size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        probs = utils.vec_to_adj_mat(probs, self.n_nodes)
        # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        return samples, states, probs

    def log_prob(self, adj_matrices, return_probs=False):
        #TODO: the computation of the log likelihood it is very likely that must be changed. It needs to compute the
        # likelihood of the whole adjacency matrix, not just the likelihood of the elements off the diagonal
        """
        Compute the log probability of the given batch of adjacency matrices.
        :param adj_matrices: batch of adjacency matrices
        :param return_probs: if True the Bernoulli success probabilities for each entry of the adjacency matrices is
                             returned as well
        :return: log probability for the batch of adjacency matrices or log probability for the batch of adjacency
                 matrices and the Bernoulli success probabilities for each entry of the adjacency matrices
        """
        # adj_matrices size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        batch_size, n_nodes1, n_nodes2 = adj_matrices.shape
        assert n_nodes1 == n_nodes2 and n_nodes1 == self.n_nodes
        value = utils.adj_mat_to_vec(adj_matrices, self.n_nodes).unsqueeze(2)
        # value size = (B,T,1) where B=batch size, T=self.n_dim_out
        state = self._get_state(batch_size)  # hidden / cell state at t=0
        # state = tuple where both elements have size (B,Z,H) where B=batch size, Z=self.n_layers, H=self.hidden_dim
        input = self._get_init_input(batch_size)  # input at t=0
        # inputs size = (B,1,1) where B=batch size
        value = torch.cat([input, value], dim=-2)
        # value size = (B,T+1,1) where B=batch size, T=self.n_dim_out
        probs, _ = self._compute_outputs_states(value, state)
        # probs size = (B,T+1,1) where B=batch size, T=self.n_dim_out
        probs = probs[:, :-1, :]
        # probs size = (B,T,1) where B=batch size, T=self.n_dim_out
        value = value[:, 1:]
        # value size = (B,T,1) where B=batch size, T=self.n_dim_out
        log_probs = torch.distributions.Bernoulli(probs=probs).log_prob(value).sum(1)
        # log_probs size = (B,1)
        if return_probs:
            probs = utils.vec_to_adj_mat(probs, self.n_nodes)
            # probs size = (B,Q,Q) where B=batch size, Q=self.n_nodes
            return log_probs, probs
        return log_probs

    def entropy(self, n_samples=10 ** 6):  # TODO this function has not been checked
        bs = 100000
        curr = 0
        ent = 0.
        while curr < n_samples:
            curr_batch_size = min(bs, n_samples - curr)
            ent -= torch.sum(self.log_prob(self.sample([curr_batch_size])))
            curr += curr_batch_size
        return ent / n_samples

    def mode(self, n_samples=1000, return_logprob=False):
        """
        Compute the mode of the adjacency matrix distribution
        :param n_samples: number of samples to draw from the distribution from which to compute the mode
        :param return_logprob: if True, the log probability associated with the mode is returned as well
        :return: mode of the adjacency matrix distribution or mode of the adjacency matrix distribution and
                 its log probability
        """
        samples = self.sample(n_samples)
        # samples size = (B,Q,Q) where B=batch size, Q=self.n_nodes
        log_probs = self.log_prob(samples)
        # log_probs size = (B,1) where B=batch size
        max_idx = torch.argmax(log_probs)
        mode_ = samples[max_idx].unsqueeze(0)
        # mode_ size = (1,Q,Q) where Q=self.n_nodes
        if return_logprob:
            log_prob = (log_probs[max_idx]).unsqueeze(0)
            # log_prob size = (1,1)
            return mode_, log_probs[max_idx]
        return mode_

    def _get_state(self, batch_size=1):
        """
        Get a batch of initial states. The initial state is just repeated n times where n is the batch size value.
        :param batch_size: batch size
        :return: batch of initial states
        """
        return self.h0.repeat(batch_size, 1, 1), self.c0.repeat(batch_size, 1, 1)

    def _get_init_input(self, batch_size):
        """
        Get a batch of the initial input. The initial input is just repeated n times where n is the batch size value.
        :param batch_size: batch size
        :return: batch of the initial input
        """
        return self._init_input_param.expand(batch_size, 1, self.n_classes)

    @staticmethod
    def _t(a):
        return [t.transpose(0, 1).contiguous() for t in a]