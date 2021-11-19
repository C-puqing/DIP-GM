# import torch.nn as nn
#
#
# from models.helper import *
# from models.affinity_layer import InnerProductWithWeightsAffinity
# from models.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
# from models.SinkhornModule.sinkhorn_layer import SinkhornNet
# from models.SinkhornModule.voting_layer import Voting
# import utils.backbone
# from utils.feature_align import feature_align
# from utils.utils import lexico_iter
#
#
# class CrossEntropyLoss(nn.Module):
#     """
#     Cross entropy loss between two permutations.
#     """
#
#     def __init__(self):
#         super(CrossEntropyLoss, self).__init__()
#
#     def forward(self, pred_perm, gt_perm, ns_src, ns_dst):
#         batch_num = pred_perm.shape[0]
#
#         pred_perm = pred_perm.to(dtype=torch.float32)
#         gt_perm = gt_perm.to(dtype=torch.float32)
#
#         assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
#         assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
#
#         loss = torch.tensor(0.).to(pred_perm.device)
#         n_sum = torch.zeros_like(loss)
#         for b in range(batch_num):
#             loss += F.binary_cross_entropy(
#                 pred_perm[b, :ns_src[b], :ns_dst[b]],
#                 gt_perm[b, :ns_src[b], :ns_dst[b]],
#                 reduction='sum')
#             n_sum += ns_src[b].to(n_sum.dtype).to(pred_perm.device)
#
#         return loss / n_sum
#
#
# class Net(utils.backbone.VGG16_bn):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024)
#         self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
#             total_num_nodes=self.message_pass_node_features.num_node_features
#         )
#         self.global_state_dim = 1024
#         self.vertex_affinity = InnerProductWithWeightsAffinity(
#             self.global_state_dim, self.message_pass_node_features.num_node_features)
#         self.edge_affinity = InnerProductWithWeightsAffinity(
#             self.global_state_dim,
#             self.build_edge_features_from_node_features.num_edge_features)
#         self.voting_layer = Voting(alpha=20)
#         self.bi_stochastic = SinkhornNet(max_iter=20, epsilon=1.0e-10)
#
#     def forward(
#             self,
#             images,
#             points,
#             graphs,
#             n_points,
#             perm_mats,
#             visualize_flag=False,
#             visualization_params=None,
#     ):
#
#
#         # return matchings
