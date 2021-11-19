import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn


class SinkhornNet(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """

    def __init__(self, max_iter=10, epsilon=1e-4):
        super(SinkhornNet, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s: torch.Tensor, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False,):
        input_s = torch.detach(s)
        batch_size = s.shape[0]

        # if dummy_row:
        #     dummy_shape = list(s.shape)
        #     dummy_shape[1] = s.shape[2] - s.shape[1]
        #     s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
        #     new_nrows = ncols
        #     for b in range(batch_size):
        #         s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
        #     nrows = new_nrows

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon

        for i in range(self.max_iter):
            if exp:
                s = torch.exp(exp_alpha * s)
            if i % 2 == 1:
                # row norm
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # column norm
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        # if dummy_row and dummy_shape[1] > 0:
        #     s = s[:, :-dummy_shape[1]]

        if not torch.all((s >= 0) * (s <= 1)).item():
            print(s)

        return s


def hungarian(s: torch.Tensor, n1=None, n2=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    for b in range(batch_num):
        n1b = perm_mat.shape[1] if n1 is None else n1[b]
        n2b = perm_mat.shape[2] if n2 is None else n2[b]
        row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        perm_mat[b] = np.zeros_like(perm_mat[b])
        perm_mat[b, row, col] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat
