import torch
import torch.nn as nn


class PowerIteration(nn.Module):
    """
    Power iteration layer implemented by myself, following the paper "Deep Learning of Graph Matching"
    Parameters:
        max_iter: The numbers of the iterations of the power iteration method.
        stop_thresh: When the different between the latest solution v* and last solution v small than
                    stop_thresh, the methods will be stop.
    """

    def __init__(self, max_iter=50, stop_thresh=2e-7):
        super(PowerIteration, self).__init__()
        self.max_iter = max_iter
        self.stop_thresh = stop_thresh

    def forward(self, M):
        """ 计算公式
        v_{k+1} = (M * v_k) / l2_norm(M * v_k)
        Input:
            M: torch.Tensors with shape [max_num_src, max_num_dst], represent affinity matrix
            num_vertices_s_batch: a list of k (batch_num) integers [num_vertices(G_i) for i=1..k]
            num_vertices_t_batch: a list of k (batch_num) integers [num_vertices(H_i) for i=1..k]
        Output: eigenvector v_{k+1}

        """
        v = torch.ones(M.shape[-1]).to(M.device)
        for i in range(self.max_iter):
            res = torch.matmul(M, v)
            v = res / torch.linalg.vector_norm(res)

        return v


class Voting(nn.Module):
    """
    Voting Layer computes a new row-stotatic matrix with softmax. A large number (alpha) is multiplied to the input
    stochastic matrix to scale up the difference.
    Parameter: value multiplied before softmax alpha
               threshold that will ignore such points while calculating displacement in pixels pixel_thresh
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """

    def __init__(self, alpha=200, pixel_thresh=None):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)  # Voting among columns
        self.pixel_thresh = pixel_thresh

    def forward(self, s):
        return self.softmax(self.alpha * s)


class SpectralMatching(nn.Module):
    def __init__(self, pi_max_iter, stop_thresh):
        super(SpectralMatching, self).__init__()
        self.power_iteration = PowerIteration(pi_max_iter, stop_thresh)

    def forward(self, affinity_matrix, num_src, num_dst) -> torch.Tensor:
        # power iteration
        v = self.power_iteration(affinity_matrix)

        # reshape to a matrix with shape [num_nodes, num_nodes]
        s0 = torch.reshape(v, [num_src, num_dst])

        try:
            assert torch.all(s0 >= 0)
        except AssertionError as err:
            print(s0)
            raise err

        return s0
