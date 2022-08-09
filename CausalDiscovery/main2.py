import argparse
import torch
import os
import numpy as np


def parse_args():
    """
    Parse command line arguments
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description='Variational Structural Causal Model Inference')
    parser.add_argument('--save_path', type=str, default='results_anneal/',
                        help='path to save result files')  # TODO: go back here
    parser.add_argument('--seed', type=int, default=10, help='seed for generating random numbers (default: 10)')  # TODO: PERFECT
    parser.add_argument('--data_seed', type=int, default=20, help='random seed for generating data (default: 20)')  # TODO: PERFECT
    parser.add_argument('n_data_samples', type=int, default=1,
                        help='number of samples to be drawn from the real Structural Causal Model (default: 1)')  # TODO: PERFECT
    parser.add_argument('--n_joint_dist_samples', type=int, default=1000,
                        help='number of samples to be drawn from the adjacency matrix and structural equation'
                             'parameters variational joint distribution (default: 1000)')  # TODO: PERFECT
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate (default: 1e-2)')  # TODO: PERFECT
    parser.add_argument('--max_gibbs_temp', type=float, default=1000.0,
                        help='maximum Gibbs temperature DAG constraint (default: 1000)')  # TODO: PERFECT
    parser.add_argument('--sparsity_factor', type=float, default=0.001,
                        help='temperature for the sparsity constraint (default: 0.001)')  # TODO: PERFECT
    parser.add_argument('--epochs', type=int, default=30000, help='number of epochs (default: 30000)')  # TODO: PERFECT
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes in the causal model')  # TODO: might be useful later
    parser.add_argument('--num_samples', type=int, default=100, help='Total number of samples in the synthetic data')  # TODO: might be useful later
    parser.add_argument('--noise_type', type=str, default='isotropic-gaussian', help='Type of noise of causal model')  # TODO: might be useful later
    parser.add_argument('--noise_sigma', type=float, default=1.0, help='Std of Noise Variables')  # TODO: might be useful later
    parser.add_argument('--theta_mu', type=float, default=2.0, help='Mean of Parameter Variables')  # TODO: might be useful later
    parser.add_argument('--theta_sigma', type=float, default=1.0, help='Std of Parameter Variables')  # TODO: might be useful later
    parser.add_argument('--data_type', type=str, default='er', help='Type of data')  # TODO: might be useful later
    parser.add_argument('--exp_edges', type=float, default=1.0, help='Expected number of edges in the random graph')  # TODO: might be useful later
    parser.add_argument('--eval_only', action='store_true', default=False, help='Perform Just Evaluation')  # TODO: might be useful later
    parser.add_argument('--anneal', action='store_true', default=False,
                        help='perform gibbs temperature exponential annealing')  # TODO: PERFECT

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, args.data_type + '_' + str(int(args.exp_edges)),
                                  str(args.num_nodes) + '_' + str(args.seed) + '_' + str(args.data_seed) + '_' + str(
                                      args.num_samples) + '_' + \
                                  str(args.sparsity_factor) + '_' + str(args.gibbs_temp))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.num_nodes == 2:
        args.exp_edges = 0.8

    args.gibbs_temp_init = 10.
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args