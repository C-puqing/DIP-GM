from torch import nn

from models.helper import init_result_matrix


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

    def __init__(self, alpha=200):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, costs_batch, ns_src_batch, ns_dst_batch):
        result = init_result_matrix(costs_batch)
        for b, (costs, ns_src, ns_dst) in enumerate(zip(costs_batch, ns_src_batch, ns_dst_batch)):
            result[b, :ns_src, :ns_dst] = self.softmax(self.alpha * costs)
        return result
