from models.helper import *
from utils.feature_align import feature_align
from utils.utils import lexico_iter

class FeatureExtract(torch.autograd.Function):

    def feature_extract(model, images, points, n_points, graphs, perm_mats):
        global_list = []
        orig_graph_list = []

        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = model.node_layers(image)
            edges = model.edge_layers(nodes)

            global_list.append(model.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            node_features = torch.cat((U, F), dim=-1)
            graph.x = node_features

            graph = model.message_pass_node_features(graph)
            orig_graph = model.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_costs_list = [
            model.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_costs_list]

        quadratic_costs_list = [
            model.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        quadratic_costs_list = [[-0.5 * x for x in quadratic_costs] for quadratic_costs in quadratic_costs_list]

        if model.training:
            unary_costs_list = [
                [
                    x + 1.0 * gt[:dim_src, :dim_tgt]  # Add margin with alpha = 1.0
                    for x, gt, dim_src, dim_tgt in zip(unary_costs, perm_mat, ns_src, ns_tgt)
                ]
                for unary_costs, perm_mat, (ns_src, ns_tgt) in zip(unary_costs_list, perm_mats, lexico_iter(n_points))
            ]

        all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]

        return unary_costs_list, quadratic_costs_list, all_edges, orig_graph_list
