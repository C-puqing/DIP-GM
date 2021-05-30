from os import error
import torch
import torch.nn.functional as F

class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


class EnergyLoss(torch.nn.Module):
    def forward(self, suggested, target, unary_costs):
        errors = torch.tensor(0., device=suggested.device)
        for pred, gt, cost in zip(suggested, target, unary_costs):
            n_src, n_dst = cost.shape[0], cost.shape[1]
            errors += torch.sum((gt[:n_src, :n_dst] - pred[:n_src, :n_dst]) * cost) / n_src
        return errors / target.shape[0]