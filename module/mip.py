import torch
import numpy as np
from gurobipy import Model
from gurobipy import GRB
from gurobipy import quicksum

import torch.nn.functional as F

def gm_solver(costs, quadratic_costs, edges_src, edges_dst, solver_params):
    V1, V2 = costs.shape[0], costs.shape[1]
    E1, E2 = edges_src.shape[0], edges_dst.shape[0]

    coeff_x = dict()
    for i in range(V1):
        for j in range(V2):
            coeff_x[(i,j)] = costs[i][j]
    coeff_y = dict()
    if E1 != 0 and E2 != 0:
        for i in range(E1):
            for j in range(V2):
                coeff_y[(i,j)] = quadratic_costs[i][j]

    model = Model("gm_solver")
    for param, value in solver_params.items():
        model.setParam(param, value)

    x = model.addVars(V1, V2, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    y = model.addVars(E1, E2, lb=0, ub=1, vtype=GRB.BINARY, name="y")

    obj = x.prod(coeff_x) + y.prod(coeff_y)
    model.setObjective(obj, GRB.MINIMIZE)

    for i in range(V1):
        expr = x.select(i,'*')
        model.addConstr(quicksum(expr) == 1, name=f"{i}-th row")
    for j in range(V2):
        expr = x.select('*', j)
        model.addConstr(quicksum(expr) == 1, name=f"{j}-th col")
    for ij in range(E1):
        i, j = edges_src[ij][0], edges_src[ij][1]
        for kl in range(E2):
            k, l = edges_dst[kl][0], edges_dst[kl][1]
            expr = [
                model.addConstr(y <= x1 * x2, name=f"({ij},{kl} edges - ({i},{j}), ({k},{l})")
                for y,x1,x2 in zip(y.select(ij,kl), x.select(i,k), x.select(j,l))
            ]

    model.optimize()

    pmat_v = np.zeros(shape=(V1,V2), dtype=np.float32)
    for indx, var in zip(x, x.select()):
        pmat_v[indx] = var.X
    pmat_v = np.around(pmat_v,1)
    # assert np.all(np.sum(pmat_v, axis=-1) <= 1) and np.all(np.sum(pmat_v, axis=-2) <= 1)

    pmat_e = np.zeros(shape=(E1,E2), dtype=np.float32)
    for indx, var in zip(y, y.select()):
        pmat_e[indx] = var.X
    pmat_e = np.around(pmat_e,1)

    return pmat_v, pmat_e

class LinearModule(torch.nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pmat_v, costs):
        margin = torch.ones_like(costs).to(costs.device)
        margin *= torch.min(costs).detach().abs()
        pos = pmat_v.ge(1)
        margin[pos] *= -1
        trans_costs = costs + margin
        output = F.softmin(trans_costs, dim=1)
        # _ = F.one_hot(torch.argmax(output, dim=1), num_classes=pmat_v.shape[0])
        # assert torch.all(torch.sum(_, dim=-1) <= 1) and torch.all(torch.sum(_, dim=-2) <= 1)
        return output

class GraphMatchingModule(torch.nn.Module):
    """
    Torch module for handling batches of Graph Matching Instances
    """
    def __init__(
        self,
        edges_left_batch,
        edges_right_batch,
        num_vertices_s_batch,
        num_vertices_t_batch,
        lambda_val,
        solver_params,
    ):
        """
        Prepares a module for a batch of k graph matching instances, i.e. instances matching graphs G_i, H_i for i=1..k

        @param edges_left_batch: a list of k torch.Tensors with shape (num_edges(G_i), 2) describing edges of G_i
        @param edges_right_batch: a list of k torch.Tensors with shape (num_edges(H_i), 2) describing edges of H_i
        @param num_vertices_s_batch: a list of k integers [num_vertices(G_i) for i=1..k]
        @param num_vertices_t_batch: a list of k integers [num_vertices(H_i) for i=1..k]
        @param lambda_val: lambda value for backpropagation by [1]
        @param solver_params: a dict of command line parameters to the solver (see solver documentation)
        """
        super().__init__()
        self.discret_to_continuous = LinearModule()
        self.edges_left_batch = edges_left_batch
        self.edges_right_batch = edges_right_batch
        self.num_vertices_s_batch = num_vertices_s_batch
        self.num_vertices_t_batch = num_vertices_t_batch
        self.params = {"lambda_val": lambda_val, "solver_params": solver_params}

    def solve(self, costs, quadratic_costs, params):
        device = costs.device

        pmat_v, pmat_e = gm_solver(
            costs=costs.cpu().detach().numpy(),
            quadratic_costs=quadratic_costs.cpu().detach().numpy(),
            edges_src=params["edges_left"].cpu().detach().numpy(),
            edges_dst=params["edges_right"].cpu().detach().numpy(),
            solver_params=params["solver_params"]
        )

        pmat_v = torch.from_numpy(pmat_v).to(device).to(torch.float32)
        pmat_e = torch.from_numpy(pmat_e).to(device).to(torch.float32)

        return pmat_v, pmat_e

    def forward(self, costs_batch, quadratic_costs_batch):
        """
        Forward pass for a batch of k graph matching instances
        @param costs_batch: torch.Tensor of shape (k, max(num_vertices(G_i)), max(num_vertices(H_i))) with zero padding
        describing the unary costs of the k instances
        @param quadratic_costs_batch: torch.Tensor of shape (k, max(num_edges(G_i)), max(num_edges(H_i))) padded with
        zeros describing the quadratic costs of the k instances.
        @param gt_costs_batch: torch.Tensor of shape (k, 1) describing the true matching unary cost
        @return: torch.Tensor of shape (k, max(num_vertices(G_i)), max(num_vertices(H_i))) with 0/1 values and
        zero padding. Captures the returned matching from the solver.
        """
        def params_generator():
            for edges_left, edges_right in zip(self.edges_left_batch, self.edges_right_batch):
                yield {"edges_left": edges_left.T, "edges_right": edges_right.T, **self.params}

        def costs_generator():
            zipped = zip(
                self.edges_left_batch,
                self.edges_right_batch,
                self.num_vertices_s_batch,
                self.num_vertices_t_batch,
                costs_batch,
                quadratic_costs_batch,
            )
            for edges_left, edges_right, num_vertices_s, num_vertices_t, costs, quadratic_costs in zipped:
                truncated_costs = costs[:num_vertices_s, :num_vertices_t]
                assert quadratic_costs.shape[0] == edges_left.shape[1], (quadratic_costs.shape, edges_left.shape)
                assert quadratic_costs.shape[1] == edges_right.shape[1], (quadratic_costs.shape, edges_right.shape)
                truncated_quadratic_costs = quadratic_costs[: edges_left.shape[-1], : edges_right.shape[-1]]
                leftover_costs = (truncated_costs.abs().sum() - costs.abs().sum()).abs()
                assert leftover_costs < 1e-5, leftover_costs
                leftover_quadratic_costs = (truncated_quadratic_costs.abs().sum() - quadratic_costs.abs().sum()).abs()
                assert leftover_quadratic_costs < 1e-5, leftover_quadratic_costs
                yield truncated_costs, truncated_quadratic_costs

        batch_size = len(costs_batch)
        max_dimension_x = max(x.shape[0] for x in costs_batch)
        max_dimension_y = max(x.shape[1] for x in costs_batch)
        result = torch.zeros(size=(batch_size, max_dimension_x, max_dimension_y)).to(costs_batch[0].device)

        for i, (params, costs, num_vertices_s, num_vertices_t) in enumerate(
            zip(params_generator(), costs_generator(), self.num_vertices_s_batch, self.num_vertices_t_batch)
        ):
            unary_cost, quadratic_costs = costs[0], costs[1]
            pmat_list = self.solve(unary_cost, quadratic_costs, params)
            S_mat = self.discret_to_continuous(pmat_list[0], unary_cost)
            result[i, :num_vertices_s, :num_vertices_t] = S_mat

        return result
