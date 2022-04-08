import utils.backbone
from models.helper import *
from models.helper import hungarian
from models.power_iteration.affinity_layer import AffinityMatrixConstructLayer
from models.power_iteration.gm_solver import SpectralMatching
from models.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from utils.config import config
from utils.feature_align import feature_align
from utils.utils import lexico_iter


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
        self.gm_solver = SpectralMatching(25, 1.0e-3)

    def forward(
            self,
            images: list,
            points: list,
            graphs: list,
            n_points: list,
            perm_mats: list,
            visualize_flag=False,
            visualization_params=None,
    ):
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
                M = self.affinity_layer(graph_list, node_feats, edge_feats, weight)

                # solved the graph matching problem
                s = self.gm_solver(M, g_1.num_nodes, g_2.num_nodes)

                Ss.append(s)

        ns_g1, ns_g2 = n_points[0], n_points[1]
        max_num_node = max(max(ns_g1), max(ns_g2))

        # reshape eigenvectors in Ss to matrix with shape of [batch_size, max_num_nodes, max_num_nodes]
        results = torch.zeros([config.BATCH_SIZE, max_num_node, max_num_node],
                              device=perm_mats[0].device, dtype=perm_mats[0].dtype)
        for i, (s, num_src, num_dst) in enumerate(zip(Ss, ns_g1, ns_g2)):
            results[i, :num_src, :num_dst] = s

        results_dict = {
            'relax_sol': results,
            'perm_sol': hungarian(results, ns_g1, ns_g2)
        }

        return results_dict
