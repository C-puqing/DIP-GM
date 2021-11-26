from torch import nn

import utils.backbone
from models.DIP.affinity_layer import InnerProductWithWeightsAffinity
from models.DIP.mip import GraphMatchingModule
from models.helper import *
from models.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from utils.config import config
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from utils.visualization import easy_visualize


class Loss(nn.Module):
    """
    HammingLoss used in "Black-box deep learning graph matching". The formulation
    defined as L(v) = v * (1 - v*) + (1 - v) * v*
    """

    def __init__(self):
        super(Loss, self).__init__()

    # @staticmethod
    # def forward(suggested, target):
    #     errors = suggested * (1.0 - target) + (1.0 - suggested) * target
    #     return errors.mean(dim=0).sum()
    #
    """
    2021.11.23 在Hamming损失函数上新增边损失
    """
    @staticmethod
    def loss_on_node(suggest, target):
        errors = suggest * (1.0 - target) + (1.0 - suggest) * target
        return errors.mean(dim=0).sum()

    def forward(self, predict_list, ground_truth_list):
        loss = torch.tensor(0, dtype=torch.float32, device=predict_list[0].device)
        # calculated hamming loss both node matching and edge matching
        for pred, gt in zip(predict_list, ground_truth_list):
            loss += self.loss_on_node(pred, gt)

        return loss


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
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.build_edge_features_from_node_features.num_edge_features)

    def feature_extract(self, images, points, n_points, graphs, perm_mats):
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
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_costs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_costs_list]

        quadratic_costs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        quadratic_costs_list = [[-0.5 * x for x in quadratic_costs] for quadratic_costs in quadratic_costs_list]

        if self.training:
            unary_costs_list = [
                [
                    x + 1.0 * gt[:dim_src, :dim_tgt]  # Add margin with alpha = 1.0
                    for x, gt, dim_src, dim_tgt in zip(unary_costs, perm_mat, ns_src, ns_tgt)
                ]
                for unary_costs, perm_mat, (ns_src, ns_tgt) in zip(unary_costs_list, perm_mats, lexico_iter(n_points))
            ]

        all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]

        return unary_costs_list, quadratic_costs_list, all_edges, orig_graph_list

    def forward(
            self,
            images: list,
            points: list,
            graphs: list,
            n_points: list,
            ground_truth_list: list,
            visualize_flag=False,
            visualization_params=None,
    ):
        # 特征提取
        unary_costs_list, quadratic_costs_list, all_edges, orig_graph_list = self.feature_extract(
            images, points, n_points, graphs, ground_truth_list)

        # 配置图匹配求解模块
        gm_solvers = [
            GraphMatchingModule(
                all_left_edges,
                all_right_edges,
                ns_src,
                ns_tgt,
                config.SOLVER_SETTING.lambda_val,
                config.SOLVER_SETTING.solver_params,
            )
            for (all_left_edges, all_right_edges), (ns_src, ns_tgt) in zip(
                lexico_iter(all_edges), lexico_iter(n_points)
            )
        ]

        matching_results = [
            gm_solver(unary_costs, quadratic_costs)
            for gm_solver, unary_costs, quadratic_costs in zip(gm_solvers, unary_costs_list, quadratic_costs_list)
        ]

        if config.VERBOSE_SETTING.visualize:
            easy_visualize(
                orig_graph_list,
                points,
                n_points,
                images,
                unary_costs_list,
                quadratic_costs_list,
                matching_results[0],
                ground_truth_list,
                **config.VERBOSE_SETTING.visualization_params,
            )

        return matching_results
