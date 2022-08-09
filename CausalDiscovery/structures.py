import numpy as np
from causalgraphicalmodels import StructuralCausalModel
import torch


class SCM(StructuralCausalModel):
    """
    Class modelling a Structural Causal Model. This class extends the StructuralCausalModel class from the
    causalgraphicalmodels class. When instantiating an instance of this class, the nodes need to be named "x{index}"
    where {index} is an integer number denoting the index of that node. The index must start from 0.
    """
    def __init__(self, assignment, device="cpu"):
        """
        Initialise Structural Causal Model
        :param assignment: structural equation assignment
        :param device: device on which to store the data sampled from the Structural Causal Model
        """
        super().__init__(assignment)
        self.device = device

    def sample(self, n_samples=100, set_values=None):
        """
        Sample a batch of data from the Structural Causal Model.
        :param n_samples: number of samples
        :param set_values:
        :return: batch of data
        """
        df = super().sample(n_samples, set_values)
        cols = df.columns.tolist()
        cols.sort()
        return torch.tensor(df[cols].to_numpy(), dtype=torch.float32, device=self.device)

    def do(self, node, f):
        """
        Intervene upon the given node by replacing its current value generating function with a new one.
        :param node: index of node upon which the intervention needs to be performed
        :param f: function generating the intervened node values given its parents
        :return: intervened Structural Causal Model
        """
        new_assignment = self.assignment.copy()
        node = "x" + str(node)
        new_assignment[node] = f
        return SCM(new_assignment, self.device)

    def __len__(self):
        """
        Return number of nodes in the Structural Causal Model
        :return: number of nodes in the Structural Causal Model
        """
        return len(self.assignment)
