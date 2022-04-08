import utils.backbone
from models.DIP.affinity_layer import InnerProductWithWeightsAffinity
from models.SH.sinkhorn_layer import Sinkhorn
from models.helper import *
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
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.gm_solver = Sinkhorn(max_iter=20, tau=0.05, epsilon=1.0e-10)

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

        node_similarity_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]
        node_similarity_list = node_similarity_list[0]

        ns_g1, ns_g2 = n_points[0], n_points[1]
        max_num_node = max(max(ns_g1), max(ns_g2))
        relax_solutions = torch.zeros([config.TRAIN.BATCH_SIZE, max_num_node, max_num_node]).cuda()
        for b, (unary_cost, n_g1, n_g2) in enumerate(zip(node_similarity_list, ns_g1, ns_g2)):
            relax_solutions[b, :n_g1, :n_g2] = self.gm_solver(unary_cost, n_g1, n_g2)

        results_dict = {
            'relax_sol': relax_solutions,
            'perm_sol': hungarian(relax_solutions, ns_g1, ns_g2)
        }

        return results_dict
