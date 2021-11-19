import torch.nn as nn
import torch.nn.functional as F

import utils.backbone
from models.PI_SK.affinity_layer import AffinityMatrixConstructLayer
from models.PI_SK.gm_solver import GMSolver
from models.SinkhornModule.sinkhorn_layer import SinkhornNet
from models.helper import *
from models.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from utils.config import config
from utils.feature_align import feature_align
from utils.utils import lexico_iter


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

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


class Net(utils.backbone.VGG16_bn):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = 1024
        self.affinity_layer = AffinityMatrixConstructLayer(
            self.global_state_dim,
            self.message_pass_node_features.num_node_features,
            self.build_edge_features_from_node_features.num_edge_features
        )
        self.gm_solver = GMSolver(
            pi_max_iter=config.SOLVER_SETTING.PI_iter_nums,
            stop_thresh=config.SOLVER_SETTING.PI_stop_thresh,
        )
        self.sinkhorn_net = SinkhornNet(
            config.SOLVER_SETTING.SK_iter_nums,
            config.SOLVER_SETTING.SK_epsilon,
        )

    @staticmethod
    def construct_node_edge_incidence_mat(batch_graphs: list):
        """
        BUILDS NODE-EDGE INCIDENCE MATRICES G AND H FROM GIVEN AFFINITY MATRIX

        Arguments:
        ----------
            - batch_graphs: List with length batch_num, describes a batch of graph
        Returns:
        --------
            - G and H: List with length batch_num, describes a batch of g, h matrices
        """
        batch_G = []
        batch_H = []
        for graph in batch_graphs:
            G = torch.zeros(graph.num_nodes, graph.num_edges)
            H = torch.zeros(graph.num_nodes, graph.num_edges)
            heads, tails = graph.edge_index[0], graph.edge_index[1]

            # G(h,i) = 1, H(t, i) = 1, only if exits a edge from node 'h' to node 't'
            for i, (h, t) in enumerate(zip(heads, tails)):
                G[h, i] = 1
                H[t, i] = 1

            batch_G.append(G)
            batch_H.append(H)

        return batch_G, batch_H

    def forward(
            self,
            images: list,
            points: list,
            graphs: list,
            n_points: list,
            perm_mats: list,
            visualize_flag=False,
            visualization_params=None,
    ) -> dict:
        """

        Return:
            - results: torch.Tensor, consist of a batch of solution of each matching issue
        """
        global_list = []
        orig_graph_list = []

        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            node_features = torch.cat((U, F), dim=-1)
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], dim=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        Ss = list()  # save the output of Power Iteration
        for (g1_list, g2_list), global_weights, perm_mat in zip(lexico_iter(orig_graph_list), global_weights_list,
                                                                perm_mats):
            for b, (g_1, g_2, weight) in enumerate(zip(g1_list, g2_list, global_weights)):
                graph_list = [g_1, g_2]
                node_feats = [g_1.x, g_2.x]
                edge_feats = [g_1.edge_attr, g_2.edge_attr]

                # computing affinity matrix
                if self.train:
                    M = self.affinity_layer(graph_list, node_feats, edge_feats, weight, perm_mat[b])
                else:
                    M = self.affinity_layer(graph_list, node_feats, edge_feats, weight)

                # solved the graph matching problem
                s = self.gm_solver(M, g_1.num_nodes, g_2.num_nodes)

                Ss.append(s)

        ns_g1, ns_g2 = n_points[0], n_points[1]
        max_num_node = max(max(ns_g1), max(ns_g2))

        # reshape eigenvectors in Ss to matrix with shape of [batch_size, max_num_nodes, max_num_nodes]
        results = torch.zeros([config.TRAIN.BATCH_SIZE, max_num_node, max_num_node],
                              device=perm_mats[0].device, dtype=perm_mats[0].dtype)
        for i, (s, num_src, num_dst) in enumerate(zip(Ss, ns_g1, ns_g2)):
            results[i, :num_src, :num_dst] = s

        # parallel process continuous solutions by Sinkhorn Network to bi-stochastic matrix
        bi_mats = self.sinkhorn_net(results, ns_g1, ns_g2, dummy_row=True)

        # get permutation matrix by hungarian algorithm
        perm_mats = hungarian(results, n_points[0], n_points[1])

        results_dict = dict()
        results_dict.update({
            "relax_sol": bi_mats,
            "perm_mat": perm_mats
        })

        return results_dict
