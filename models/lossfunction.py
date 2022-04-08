import torch
import torch.nn as nn
import torch.nn.functional as F


class HammingLoss(nn.Module):
    """
    HammingLoss used in "Black-box deep learning graph matching". The formulation
    defined as L(v) = v * (1 - v*) + (1 - v) * v*
    """

    def __init__(self):
        super(HammingLoss, self).__init__()

    @staticmethod
    def forward(suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


class PermutationLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """

    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, ns_src, ns_dst):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)
        gt_perm = gt_perm.to(dtype=torch.float32)

        try:
            assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_perm)
            raise err

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :ns_src[b], :ns_dst[b]],
                gt_perm[b, :ns_src[b], :ns_dst[b]],
                reduction='sum')
            n_sum += ns_src[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum
