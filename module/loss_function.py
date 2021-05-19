import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def BCE(self, y_pred, y):
        return -(y * torch.log(y_pred)).sum()

    def forward(self, batch_pred_perm, batch_gt_perm, n_src, n_dst):
        ''' Calculated binary cross entropy between a batch of k graphs and ground-truths
        @params batch_pred_perm: torch.Tensor of shape [k, max(num_vertices(G_i)), max(num_vertices(H_i))] 
        with zero padding describing predict permutation matrix of k graphs
        @params batch_gt_perm: torch.Tensor of shape [k, max(num_vertices(G_i)), max(num_vertices(H_i))] 
        with zero padding describing ground-truth permutation matrix of k graphs
        @params n_src: torch.Tensor of shape [k] describing numbers of graph G_i vertices
        @params n_dst: torch.Tensor of shape [k] describing numbers of graph H_i vertices
        @return: a scalar value of loss function
        '''
        device = batch_pred_perm[0].device

        loss = torch.tensor(0.).to(device)
        for pred, gt, n_s, n_d in zip (batch_pred_perm, batch_gt_perm, n_src, n_dst):
            # loss += F.binary_cross_entropy(pred[:n_s, :n_d], gt[:n_s, :n_d], reduction="sum") / n_s
            loss += self.BCE(pred[:n_s, :n_d], gt[:n_s, :n_d]) / n_s

        # return loss / batch_gt_perm.shape[0]
        return loss