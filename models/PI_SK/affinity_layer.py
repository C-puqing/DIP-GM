import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProductAffinity(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InnerProductAffinity, self).__init__()
        self.dim = output_dim
        self.A = nn.Linear(input_dim, output_dim)

    def forward(self, X, Y, global_weight):
        assert X.shape[1] == Y.shape[1] == self.dim, (X.shape[1], Y.shape[1], self.dim)
        assert X.device == Y.device == global_weight.device
        coefficients = torch.tanh(self.A(global_weight))
        aff_mat = torch.matmul(X * coefficients, Y.transpose(0, 1))
        aff_mat = F.softplus(aff_mat) - 0.5

        return aff_mat


class AffinityMatrixConstructLayer(nn.Module):
    def __init__(self, global_dim, node_output_dim, edge_output_dim):
        super(AffinityMatrixConstructLayer, self).__init__()
        # parameter shape [1024, 1024]
        self.vertex_affinity_layer = InnerProductAffinity(global_dim, node_output_dim)
        self.edge_affinity_layer = InnerProductAffinity(global_dim, edge_output_dim)

    @staticmethod
    def construct_edge_incidence_mat(graph):
        num_nodes, num_edges = graph.num_nodes, graph.num_edges
        G = torch.zeros(num_nodes, num_edges)
        H = torch.zeros(num_nodes, num_edges)

        heads, tails = graph.edge_index[0], graph.edge_index[1]
        for i, (h, t) in enumerate(zip(heads, tails)):
            G[h, i] = 1
            H[t, i] = 1

        return G, H

    @staticmethod
    def refactor_affinity_matrix(Mp, Me, G1, G2, H1, H2):
        """
        Refactor affinity matrix by factorize origin input node-similarity matrix and edge-similarity matrix.
        M = [vec(Mp)] + (G2 opt(kron) G1)[vec(Me)](H2 opt(kron) H1)^T
        """
        Kg = torch.kron(G2, G1).to(Mp.device)
        Kh = torch.kron(H2, H1).to(Mp.device)

        total_num_Mp = torch.numel(Mp)
        vec_Mp = torch.reshape(Mp, (total_num_Mp, 1)).to(Mp.device)
        diagonal_mat_Mp = torch.eye(total_num_Mp, total_num_Mp).to(Mp.device)
        # diagonal_mat_Mp.cuda()

        total_num_Me = torch.numel(Me)
        vec_Me = torch.reshape(Me, (total_num_Me, 1)).to(Me.device)
        diagonal_mat_Me = torch.eye(total_num_Me, total_num_Me).to(Me.device)
        # diagonal_mat_Me.cuda()

        refactor_Me = torch.matmul(diagonal_mat_Me * vec_Me, Kh.T)
        refactor_Me = torch.matmul(Kg, refactor_Me)
        M = diagonal_mat_Mp * vec_Mp + refactor_Me

        return M

    def forward(self, graph_list, node_feats, edge_feats, global_weight, perm_mat=None):
        """
        Computed the affinity matrix M according the function

        Arguments:
            graph_list: a list consisting of matching graph, incorporate src graph and dst graph
            node_feats: a list consisting of src graph node features and dst graph node features
            edge_feats: a list consisting of src graph edge features and dst graph edge features
            global_weight: a list consisting of global_weight belong to src graph and dst graph
            perm_mat:

        Return:
            M: torch.Tensor, represented affinity matrix of matching issue
        """
        # calculated vertex affinity
        Mp = self.vertex_affinity_layer(node_feats[0], node_feats[1], global_weight)
        # if self.training:
        #     ns_src, ns_dst = node_feats[0].shape[0], node_feats[1].shape[0]
        #     Mp = Mp + 1.0 * perm_mat[:ns_src, :ns_dst]
        Mp = torch.relu(Mp)

        # calculated edge affinity
        Me = self.edge_affinity_layer(edge_feats[0], edge_feats[1], global_weight)
        Me = torch.relu(Me)

        # computed G1, G2, H1, H2
        G1, H1 = self.construct_edge_incidence_mat(graph_list[0])
        G2, H2 = self.construct_edge_incidence_mat(graph_list[1])

        # combined vertex affinity and edge affinity to a big affinity matrix M
        M = self.refactor_affinity_matrix(Mp, Me, G1, G2, H1, H2)

        return M
