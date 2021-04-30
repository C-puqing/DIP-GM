import torch.nn as nn
import torch
import numpy as np
from gurobipy import *

'''
MIPFocus: 1——找可行解，2——找最优解
ConcurrentMIP
'''

# mip_params = {
#     "LogToConsole":0,
#     "LogFile":"results/voc_all_keypoints/mip_log.log",
#     "MIPFocus":1,
#     "SolutionLimit":1
#     "Method":4,
# }

def gm_solver(costs, quadratic_costs, edges_src, edges_dst, solver_params):
    
    V1, V2 = costs.shape[0], costs.shape[1]
    E1, E2 = edges_src.shape[0], edges_dst.shape[0]

    model = Model("gm_solver")
    for param, value in solver_params.items():
        model.setParam(param, value)

    # 定义输入变量
    x = model.addMVar(shape=V1*V2, vtype=GRB.BINARY, name="x")
    y = model.addMVar(shape=E1*E2, vtype=GRB.BINARY, name="y")

    costs = costs.flatten()
    quadratic_costs = quadratic_costs.flatten()

    for i in range(V1):
        idx = i*V2
        model.addConstr(x[idx:idx+V2].sum() <= 1, name="row"+str(i))
    for i in range(V2):
        x_col = [x[j*V2+i] for j in range(V1)]
        model.addConstr(sum(x_col) <= 1, name="col"+str(i))
        
    for ij in range(E1):
        i, j = edges_src[ij][0], edges_src[ij][1]
        for k in range(V2):
            y_ik, y_jl = 0, 0
            for r in range(E2):
                # 头节点为 k
                if edges_dst[r][0] == k:
                    y_ik += y[ij*E2+r]
                # 尾节点为 k
                elif edges_dst[r][1] == k:
                    y_jl += y[ij*E2+r]
            model.addConstr(y_ik <= x[i*V2+k], name="")
            model.addConstr(y_jl <= x[j*V2+k], name="")

    obj = costs @ x + quadratic_costs @ y

    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    costs_paid = np.array(x.X, dtype=np.long)
    costs_paid.resize((V1,V2))
    quadratic_costs_paid = np.array(y.X, dtype=np.long)
    quadratic_costs_paid.resize((E1,E2))

    return costs_paid, quadratic_costs_paid


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
    def backward(ctx, grad_costs_paid, grad_quadratic_costs_paid, ):
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

        # 插值函数 x' = x + lambda * grad_x
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

        # f_lambda 函数的梯度 -1/lambda * [y(w) - y_lambda(w)]
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
            grad_costs, grad_quadratic_costs = self.solver.apply(costs[0], costs[1], params)
            result[i, :num_vertices_s, :num_vertices_t] = grad_costs  # Only unary matching returned

        return result
