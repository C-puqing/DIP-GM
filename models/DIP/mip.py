import numpy as np
import torch
from gurobipy import GRB
from gurobipy import Model
from gurobipy import quicksum

from models.helper import init_result_matrix


def gm_solver(costs, quadratic_costs, edges_src, edges_dst, solver_params):
    # print("costs matrix shape: ", costs.shape)
    V1, V2 = costs.shape[0], costs.shape[1]
    E1, E2 = edges_src.shape[0], edges_dst.shape[0]

    coefficient_x = dict()
    for i in range(V1):
        for j in range(V2):
            coefficient_x[(i, j)] = costs[i][j]
    coefficient_y = dict()
    if E1 != 0 and E2 != 0:
        for i in range(E1):
            for j in range(V2):
                coefficient_y[(i, j)] = quadratic_costs[i][j]

    model = Model("gm_solver")
    for param, value in solver_params.items():
        model.setParam(param, value)

    x = model.addVars(V1, V2, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    y = model.addVars(E1, E2, lb=0, ub=1, vtype=GRB.BINARY, name="y")

    obj = x.prod(coefficient_x) + y.prod(coefficient_y)
    model.setObjective(obj, GRB.MINIMIZE)

    # 顶点的行约束
    [model.addConstr(quicksum(x.select(i, '*')) <= 1) for i in range(V1)]
    # 顶点的列约束
    [model.addConstr(quicksum(x.select('*', j)) <= 1) for j in range(V2)]
    # 边的行约束
    [model.addConstr(quicksum(y.select(ij, '*')) <= 1) for ij in range(E1)]
    # 边的列约束
    [model.addConstr(quicksum(y.select('*', kl)) <= 1) for kl in range(E2)]

    beg = 0
    for ij in range(E1):
        i = edges_src[ij][0]
        ls = []
        for k in range(V2):
            for kl in range(beg, E2):
                if edges_dst[kl][0] == k:
                    ls.append(y.select(ij, kl)[0])
                else:
                    break;
                beg += 1
            model.addConstr(quicksum(ls) <= x.select(i, k)[0])

    tp_tail_constrs = [
        model.addConstr(
            quicksum([y.select(ij, kl)[0] for kl in range(E2) if edges_dst[kl][1] == l]) <=
            x.select(edges_src[ij][1], l)[0]
        )
        for ij in range(E1) for l in range(V2)
    ]

    model.optimize()

    assert model.status == GRB.OPTIMAL

    permutation_matrix_x = np.zeros(shape=(V1, V2), dtype=np.long)
    for index, var in zip(x, x.select()):
        permutation_matrix_x[index] = var.X

    permutation_matrix_y = np.zeros(shape=(E1, E2), dtype=np.long)
    for index, var in zip(y, y.select()):
        permutation_matrix_y[index] = var.X

    return permutation_matrix_x, permutation_matrix_y


class GraphMatchingSolver(torch.autograd.Function):
    """
    Graph Matching solver as a torch.Function where the backward pass is provided by
    [1] 'Vlastelica* M, Paulus* A., Differentiation of Blackbox Combinatorial Solvers, ICLR 2020'
    """

    @staticmethod
    def forward(ctx, costs, quadratic_costs, params):
        """
        Implementation of the forward pass of min-cost matching between two directed graphs
        G_1 = (V_1, E_1), G_2 = (V_2, E_2)
        @param ctx: context for backpropagation
        @param costs: torch.Tensor of shape (|V_1|, |V_2|) with unary costs of the matching instance
        @param quadratic_costs: torch.Tensor of shape (|E_1|, |E_2|) with pairwise costs of the matching instance
        @param gt_cost: torch.Tensor of shape(1) decribed the truth unary cost value
        @param params: a dict of additional params. Must contain:
                edges_src: a torch.tensor os shape (|E_1|, 2) describing edges of G_1,
                edges_dst: a torch.tensor os shape (|E_2|, 2) describing edges of G_2.
                lambda_val: float/np.float32/torch.float32, the value of lambda for computing the gradient with [1]
                solver_params: a dict of command line parameters to the solver (see solver documentation)
        @return: torch.Tensor of shape (|V_1|, |V_2|) with 0/1 values capturing the suggested min-cost matching
                 torch.Tensor of shape (|E_1|, |E_2|) with 0/1 values capturing which pairwise costs were paid in
                 the suggested matching
        """
        device = costs.device

        costs_paid, quadratic_costs_paid = gm_solver(
            costs=costs.cpu().detach().numpy(),
            quadratic_costs=quadratic_costs.cpu().detach().numpy(),
            edges_src=params["edges_left"].cpu().detach().numpy(),
            edges_dst=params["edges_right"].cpu().detach().numpy(),
            solver_params=params["solver_params"]
        )
        costs_paid = torch.from_numpy(costs_paid).to(torch.float32).to(device)
        quadratic_costs_paid = torch.from_numpy(quadratic_costs_paid).to(torch.float32).to(device)

        ctx.params = params
        ctx.save_for_backward(costs, costs_paid, quadratic_costs, quadratic_costs_paid)

        return costs_paid, quadratic_costs_paid

    @staticmethod
    def backward(ctx, grad_costs_paid, grad_quadratic_costs_paid):
        """
        Backward pass computation.
        @param ctx: context from the forward pass
        @param grad_costs_paid: "dL / d costs_paid" torch.Tensor of shape (|V_1|, |V_2|)
        @param grad_quadratic_costs_paid: "dL / d quadratic_costs_paid" torch.Tensor of shape (|E_1|, |E_2|)
        @return: gradient dL / costs, dL / quadratic_costs
        """
        costs, costs_paid, quadratic_costs, quadratic_costs_paid = ctx.saved_tensors
        device = costs.device
        lambda_val = ctx.params["lambda_val"]
        epsilon_val = 1e-8
        assert grad_costs_paid.shape == costs.shape and grad_quadratic_costs_paid.shape == quadratic_costs.shape

        # x' = x + lambda * grad_x
        costs_prime = costs + lambda_val * grad_costs_paid
        quadratic_costs_prime = quadratic_costs + lambda_val * grad_quadratic_costs_paid

        costs_paid_prime, quadratic_costs_paid_prime = gm_solver(
            costs=costs_prime.cpu().detach().numpy(),
            quadratic_costs=quadratic_costs_prime.cpu().detach().numpy(),
            edges_src=ctx.params["edges_left"].cpu().detach().numpy(),
            edges_dst=ctx.params["edges_right"].cpu().detach().numpy(),
            solver_params=ctx.params["solver_params"]
        )
        costs_paid_prime = torch.from_numpy(costs_paid_prime).to(torch.float32).to(device)
        quadratic_costs_paid_prime = torch.from_numpy(quadratic_costs_paid_prime).to(torch.float32).to(device)

        grad_costs = -(costs_paid - costs_paid_prime) / (lambda_val + epsilon_val)
        grad_quadratic_costs = -(quadratic_costs_paid - quadratic_costs_paid_prime) / (lambda_val + epsilon_val)

        return grad_costs, grad_quadratic_costs, None


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
        self.solver = GraphMatchingSolver()
        self.edges_left_batch = edges_left_batch
        self.edges_right_batch = edges_right_batch
        self.num_vertices_s_batch = num_vertices_s_batch
        self.num_vertices_t_batch = num_vertices_t_batch
        self.params = {"lambda_val": lambda_val, "solver_params": solver_params}

    def forward(self, costs_batch, quadratic_costs_batch):
        """
        Forward pass for a batch of k graph matching instances
        @param costs_batch: torch.Tensor of shape (k, max(num_vertices(G_i)), max(num_vertices(H_i))) with zero padding
        describing the unary costs of the k instances
        @param quadratic_costs_batch: torch.Tensor of shape (k, max(num_edges(G_i)), max(num_edges(H_i))) padded with
        zeros describing the quadratic costs of the k instances.
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

        result = init_result_matrix(costs_batch)
        for i, (params, costs, num_vertices_s, num_vertices_t) in enumerate(
                zip(params_generator(), costs_generator(), self.num_vertices_s_batch, self.num_vertices_t_batch)
        ):
            solutions = self.solver.apply(costs[0], costs[1], params)
            result[i, :num_vertices_s, :num_vertices_t] = solutions[0]  # Only unary matching result returned

        return result
