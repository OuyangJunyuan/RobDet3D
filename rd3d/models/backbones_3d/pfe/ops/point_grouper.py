import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .builder import grouper, querier
from .....utils.common_utils import ScopeTimer, gather, apply1d


class PointGrouping(nn.Module):
    def __init__(self, grouping_cfg, input_channels):
        super(PointGrouping, self).__init__()
        self.cfg = grouping_cfg
        self.need_query_features = False
        self.use_xyz = self.cfg.get('xyz', True)

        self.output_channels = self.cfg.mlps[-1]
        self.input_channels = input_channels + 3 if self.use_xyz else input_channels
        self.mlps = self.build_mlps(self.cfg.mlps, in_channels=self.input_channels)

    @staticmethod
    def build_mlps(mlp_cfg, in_channels):
        shared_mlp = []
        for k in range(len(mlp_cfg)):
            shared_mlp.extend([
                nn.Linear(in_channels, mlp_cfg[k], bias=False),
                nn.BatchNorm1d(mlp_cfg[k]),
                nn.ReLU()
            ])
            in_channels = mlp_cfg[k]
        return nn.Sequential(*shared_mlp)


@grouper.register_module('all')
class AllPointGrouping(PointGrouping):
    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(AllPointGrouping, self).__init__(grouping_cfg, input_channels)

    def group(self, xyz: torch.Tensor, features: torch.Tensor = None, *args, **kwargs):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param features: (B, N, C) descriptors of the features
        :return:
            new_features: (B, 1, N, C{ + 3})
        """
        grouped_xyz = xyz.unsqueeze(1)  # (B,1,N,3)
        if features is not None:
            grouped_features = features.unsqueeze(1)  # (B,1,N,C)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=-1)  # (B,1,N,3+C)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features,

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None, *args, **kwargs):
        """
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param new_xyz: not used
             :param features: (B, N, C) descriptors of the features
             :return:
                 new_features: (B,  3 + C, npoint, nsample)
        """
        new_features = self.group(xyz, features)  # (B,1,N,C1)
        new_features = apply1d(self.mlps, new_features)  # (B,1,N,C2)
        new_features = torch.max(new_features, dim=2)[0]  # (B,1,C2)
        return new_features


@grouper.register_module('ball')
class BallQueryPointGrouping(PointGrouping):
    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(BallQueryPointGrouping, self).__init__(grouping_cfg, input_channels)

        radius = self.cfg.query.radius
        self.radius = [radius] if isinstance(radius, float) else radius
        self.neighbour = self.cfg.query.neighbour
        querier_type = 'ball_dilated' if self.radius.__len__() > 1 else 'ball'
        self.ctx = {}
        if "ctx" in kwargs and kwargs["ctx"] and hasattr(kwargs["ctx"][0], "ctx"):
            querier_type = "grid_ball"
            self.ctx = kwargs["ctx"][0].ctx
        self.querier = partial(querier.from_name(querier_type), *self.radius, self.neighbour)

    def query_and_group(self, new_xyz: torch.Tensor, xyz: torch.Tensor, feats: torch.Tensor = None):
        """
         :param new_xyz:    (B, M, 3) centroids
         :param xyz:        (B, N, 3) xyz coordinates of the features
         :param feats:      (B, N, C) descriptors of the features
         :return:
             empty_mask:    (B, M) tensor with the number of grouped points for each ball query
             new_feats:     (B, M, K, {3 +} C)
        """
        if self.ctx:
            group_member_cnt, group_member_ind = self.querier(xyz, new_xyz,
                                                              self.ctx["voxels"],
                                                              self.ctx["voxel_hashes"],
                                                              self.ctx["hash2query"])
        else:
            group_member_cnt, group_member_ind = self.querier(xyz, new_xyz)  # (B,M) (B,M,K)
        empty_mask = group_member_cnt > 0

        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)

        if feats is not None:
            feats = gather(feats, group_member_ind)
            new_feats = torch.cat([grouped_xyz, feats], dim=-1) if self.use_xyz else feats  # (B,M,K,{3+}C)
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_feats = grouped_xyz

        return empty_mask, new_feats

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        empty_mask, new_feats = self.query_and_group(new_xyz, xyz, feats)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None, None]  # (B,M,K,C1)
        new_feats = torch.max(new_feats, dim=2)[0]  # (B,M,Co)
        return new_feats,


@grouper.register_module('fast-ball')
class FastBallQueryPointGrouping(PointGrouping):
    """
    accelerate mlp for efficient inference.
    Reference Paper: https://ojs.aaai.org/index.php/AAAI/article/view/16207
    Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection
    """

    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(FastBallQueryPointGrouping, self).__init__(grouping_cfg, input_channels)

        radius = self.cfg.query.radius
        self.radius = [radius] if isinstance(radius, float) else radius
        self.neighbour = self.cfg.query.neighbour
        querier_type = 'ball_dilated' if self.radius.__len__() > 1 else 'ball'
        self.querier = partial(querier.from_name(querier_type), *self.radius, self.neighbour)

    def build_mlps(self, mlp_cfg, in_channels):
        build_mlps_func = super(FastBallQueryPointGrouping, self).build_mlps
        self.add_module('xyz_encoder', build_mlps_func(mlp_cfg[:1], 3)[:2])
        self.add_module('feat_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        self.add_module('nonlinear', nn.ReLU())
        refine_encoder = build_mlps_func(mlp_cfg[1:], mlp_cfg[0])
        return refine_encoder

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        feat_feats = apply1d(self.feat_encoder, feats)  # (B,M,C1)

        group_member_cnt, group_member_ind = self.querier(xyz, new_xyz)  # (B,M) (B,M,K)
        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)

        empty_mask = group_member_cnt > 0
        grouped_feat_feats = gather(feat_feats, group_member_ind)  # (B,M,K,C1)
        grouped_xyz_feats = apply1d(self.xyz_encoder, grouped_xyz)  # (B,M,K,C1)
        grouped_feats = self.nonlinear(grouped_xyz_feats + grouped_feat_feats)  # (B,M,K,C1)

        new_feats = torch.max(grouped_feats, dim=2)[0]  # (B,M,C1)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None]  # (B,M,C2)
        return new_feats,


@grouper.register_module('norm-fast-ball')
class NormalizedFastBallQueryPointGrouping(PointGrouping):
    """
    add GEOMETRIC AFFINE MODULE (layer-norm in point cloud) into ball group.
    Reference Paper: https://arxiv.org/abs/1907.03670
    RETHINKING NETWORK DESIGN AND LOCAL GEOMETRY IN POINT CLOUD: A SIMPLE RESIDUAL MLP FRAMEWORK.

    Warnings: seem to have a bad performance.
    """

    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(NormalizedFastBallQueryPointGrouping, self).__init__(grouping_cfg, input_channels)

        radius = self.cfg.query.radius
        self.radius = [radius] if isinstance(radius, float) else radius
        self.neighbour = self.cfg.query.neighbour
        querier_type = 'ball_dilated' if self.radius.__len__() > 1 else 'ball'
        self.querier = partial(querier.from_name(querier_type), *self.radius, self.neighbour)

        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, self.cfg.mlps[0]]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, self.cfg.mlps[0]]))

    def build_mlps(self, mlp_cfg, in_channels):
        build_mlps_func = super(NormalizedFastBallQueryPointGrouping, self).build_mlps
        self.add_module('xyz_encoder', build_mlps_func(mlp_cfg[:1], 3)[:2])
        self.add_module('feat_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        self.add_module('nonlinear', nn.ReLU())
        refine_encoder = build_mlps_func(mlp_cfg[1:], mlp_cfg[0])
        return refine_encoder

    def normalized(self, feats):
        center = feats[:, :, 0:1, :]
        relative = feats - center
        std = relative.view(feats.shape[0], -1).std(dim=-1)[..., None, None, None]
        feats = relative / (std + 1e-5)
        feats = self.affine_alpha * feats + self.affine_beta
        return feats

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        feat_feats = apply1d(self.feat_encoder, feats)  # (B,M,C1)

        group_member_cnt, group_member_ind = self.querier(xyz, new_xyz)  # (B,M) (B,M,K)
        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)

        empty_mask = group_member_cnt > 0
        grouped_feat_feats = gather(feat_feats, group_member_ind)  # (B,M,K,C1)
        grouped_xyz_feats = apply1d(self.xyz_encoder, grouped_xyz)  # (B,M,K,C1)
        grouped_feats = self.nonlinear(grouped_xyz_feats + grouped_feat_feats)  # (B,M,K,C1)
        grouped_feats = self.normalized(grouped_feats)

        new_feats = torch.max(grouped_feats, dim=2)[0]  # (B,M,C1)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None]  # (B,M,C2)
        return new_feats,


@grouper.register_module('point-transformer')
class PointTransformer(PointGrouping):
    """
    local-attention point transformer
    Reference Paper: https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html
    Point Transformer
    """

    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(PointTransformer, self).__init__(grouping_cfg, input_channels)

        radius = self.cfg.query.radius
        self.radius = [radius] if isinstance(radius, float) else radius
        self.neighbour = self.cfg.query.neighbour
        querier_type = 'ball_dilated' if self.radius.__len__() > 1 else 'ball'
        self.querier = partial(querier.from_name(querier_type), *self.radius, self.neighbour)

    def build_mlps(self, mlp_cfg, in_channels):
        build_mlps_func = super(PointTransformer, self).build_mlps
        self.add_module('xyz_encoder', build_mlps_func(mlp_cfg[:1], 3)[:2])
        self.add_module('feat_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        self.add_module('attention_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        refine_encoder = build_mlps_func(mlp_cfg[1:], mlp_cfg[0])
        return refine_encoder

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        feat_feats = apply1d(self.feat_encoder, feats)  # (B,M,C1)
        attention = apply1d(self.attention_encoder, feats)  # (B,M,C1)

        group_member_cnt, group_member_ind = self.querier(xyz, new_xyz)  # (B,M) (B,M,K)
        empty_mask = group_member_cnt > 0

        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)
        grouped_feat_feats = gather(feat_feats, group_member_ind)  # (B,M,K,C1)
        grouped_attention = gather(attention, group_member_ind)  # (B,M,K,C1)

        grouped_xyz_feats = apply1d(self.xyz_encoder, grouped_xyz)  # (B,M,K,C1)
        grouped_feat_feats += grouped_xyz_feats  # (B,M,K,C1)
        grouped_attention += grouped_xyz_feats  # (B,M,K,C1)

        new_feats = (grouped_feat_feats.softmax(dim=2) * grouped_attention).sum(dim=2)  # (B,M,C1)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None]  # (B,M,C2)
        return new_feats,

# @grouper.register_module('dbq', 'dyn_ball')
# class DynamicBallQueryPointGrouping(PointGrouping):
#     shared_dict = {}
#
#     def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
#         super(DynamicBallQueryPointGrouping, self).__init__(grouping_cfg, input_channels)
#         self.need_query_features = True
#         self.pred_mask = nn.Conv1d(input_channels, 1, kernel_size=1, bias=True)
#
#         self.neighbour, self.radius = self.cfg.query.neighbour, self.cfg.query.radius
#         self.querier = partial(querier.from_name('ball_stack'), self.radius, self.neighbour)
#
#         self.tau = 1
#         self.gamma = 0
#
#         self.a = self.cfg.train.loss.a
#         self.b = self.cfg.train.loss.b
#         self.tag = self.cfg.train.loss.tb_tag
#         self.weight = self.cfg.get('weight', 0.1)
#
#         self.init_weights()
#
#     def init_weights(self):
#         nn.init.xavier_normal_(self.pred_mask.weight)
#         nn.init.constant_(self.pred_mask.bias, 10)
#
#     @staticmethod
#     def build_mlps(mlp_cfg, in_channels):
#         shared_mlp = []
#         for k in range(len(mlp_cfg)):
#             shared_mlp.extend([
#                 nn.Conv1d(in_channels, mlp_cfg[k], kernel_size=1, bias=False),
#                 nn.BatchNorm1d(mlp_cfg[k]),
#                 nn.ReLU()
#             ])
#             in_channels = mlp_cfg[k]
#         return nn.Sequential(*shared_mlp)
#
#     @staticmethod
#     def __annealing(pct, start=1.0, end=0.0):
#         "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
#         cos_out = torch.cos(3.1415926 * pct) + 1  # 2->0
#         return end + (start - end) / 2 * cos_out
#
#     @staticmethod
#     def __step(x):
#         return (x > 0).float()
#
#     @staticmethod
#     def __gumble_sigmoid(logits, tau):
#         # gumbels ~ Gumbel(0,1)
#         gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
#         gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
#         gumble_sigmoid = gumbels.sigmoid()
#         # eps = 1e-20
#         # u = torch.rand_like(logits)
#         # gumbels = -torch.log(-torch.log(u + eps) + eps)
#         # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
#         # gumble_sigmoid = gumbels.sigmoid()
#         return gumble_sigmoid
#
#     def __dbq(self, xyz: torch.Tensor, new_xyz: torch.Tensor,
#               feats: torch.Tensor, new_feats: torch.Tensor,
#               batch_id: torch.Tensor, new_batch_id: torch.Tensor):
#
#         (b, c1, n), (b, c2, m) = feats.size(), new_feats.size()
#         xyz_cnt = xyz.new_tensor(b * [n], dtype=torch.int32)  # (B)
#
#         """ predict mask for new_xyz """
#         # with TimeMeasurement(f"Mask-{xyz.shape[1]}-{new_xyz.shape[1]}-{self.neighbour}: "):
#         gating_logit = self.pred_mask(new_feats).squeeze(dim=1)  # gating_logit(B, M)
#         mask_float = self.__step(gating_logit) if not hasattr(self, 'mask') else self.mask  # non-differentiable
#
#         if self.training:
#             mask_soft = self.__gumble_sigmoid(gating_logit, self.tau)  # differentiable
#             mask_float = (mask_float - mask_soft).detach() + mask_soft  # straight through strategy.
#
#         mask = mask_float.bool()  # (B, M)
#         active_xyz_cnt = torch.sum(mask, dim=-1, dtype=torch.int32)
#         if torch.sum(active_xyz_cnt) == 0:  # not active site in new_xyz
#             return None, mask_float, mask
#
#         # TODO: bottleneck in stack query
#         active_xyz = new_xyz[mask]
#         active_batch_id = new_batch_id[mask]
#         # with TimeMeasurement(f"Stack Query-{xyz.shape[1]}-{new_xyz.shape[1]}-{self.neighbour}: "):
#         _, ind = self.querier(xyz, xyz_cnt, active_xyz, active_xyz_cnt)  # (active, K)
#         active_xyz_grouped = torch_gather(xyz, ind, active_batch_id.view(-1), channel_first=False)
#         active_xyz_grouped = (active_xyz_grouped - active_xyz.unsqueeze(1)).permute(0, 2, 1)  # (active,3,neighbour)
#
#         if feats is not None:
#             active_feats_grouped = torch_gather(feats, ind, active_batch_id.view(-1), channel_first=True)
#             active_feats_grouped = torch.cat([active_xyz_grouped, active_feats_grouped], dim=1) \
#                 if self.use_xyz else active_feats_grouped
#         else:
#             assert self.use_xyz
#             active_feats_grouped = active_xyz_grouped
#
#         return active_feats_grouped, mask_float, mask
#
#     def __try_get_loss_for_all_dbq(self):
#         loss_budget, tb_dict = 0.0, {}
#         loss_type = self.cfg.train.loss.get('name', 'time')
#         if (self.id + 1) == len(self.shared_dict):
#             budget_map = lambda x, p1, p2: x * p1 + p2
#             if loss_type == 'time':
#                 budget_active, budget_all = 0, 0
#                 upper, lower = 0, 0
#                 num_layer = len(self.shared_dict)
#                 p = self.__annealing(self.percentage, start=1.0, end=0.0)
#                 for k, (mask, a, b) in self.shared_dict.items():
#                     num_active = mask.sum()
#                     num_all = mask.shape[0] * mask.shape[1]
#                     ratio = num_active / num_all
#
#                     upper += torch.clamp_min(ratio - (1 - p * (1 - self.gamma)), 0) ** 2 / num_layer
#                     lower += torch.clamp_min(p * self.gamma - ratio, 0) ** 2 / num_layer
#
#                     budget_active += budget_map(num_active, a, b)
#                     budget_all += budget_map(num_all, a, b)
#                 loss_latency = (budget_active / budget_all - self.gamma) ** 2
#                 loss_budget = loss_latency + upper + lower
#                 tb_dict.update({'dbq_loss_latency': loss_latency.item(),
#                                 'dbq_loss_lower': lower.item(),
#                                 'dbq_loss_upper': upper.item()})
#
#             elif loss_type == 'anneal':
#                 budget_active, budget_all, lower = 0, 0, 0
#                 target = self.__annealing(self.percentage, start=1.0, end=self.gamma)
#                 for k, (mask, a, b) in self.shared_dict.items():
#                     num_active = mask.sum()
#                     num_all = mask.shape[0] * mask.shape[1]
#                     budget_active += budget_map(num_active, a, b)
#                     budget_all += budget_map(num_all, a, b)
#                     lower += torch.clamp_min(target - (num_active / num_all), 0) ** 2
#                     # loss_latency = torch.nn.functional.smooth_l1_loss(budget_active / budget_all, target)
#                 loss_latency = torch.abs(budget_active / budget_all - target)
#                 loss_budget = loss_latency + lower
#                 tb_dict.update({'dbq_loss_latency': loss_latency.item(),
#                                 'dbq_loss_lower': lower.item(),
#                                 'budget_target': target})
#
#             elif loss_type == "more_one_group":
#                 layers = {n.item(): [] for n in torch.tensor([v[0].shape[1] for k, v in self.shared_dict.items()])}
#                 [layers[v[0].shape[1]].append(v) for k, v in self.shared_dict.items()]
#
#                 budget_active, budget_all, punish = 0., 0., 0.
#                 for k, layer in layers.items():
#                     mask_list, a_list, b_list = [list(item) for item in zip(*layer)]
#                     num_all = 0
#                     for mask, a, b, in zip(mask_list, a_list, b_list):
#                         num_active = mask.sum()
#                         num_all = mask.shape[0] * mask.shape[1]
#                         budget_active += budget_map(num_active, a, b)
#                         budget_all += budget_map(num_all, a, b)
#
#                     masks = torch.hstack(mask_list).view(*mask_list[0].shape, -1)
#                     kill_ratio = 0.5 - torch.clamp_max(
#                         torch.clamp_max(torch.sum(masks, dim=-1), max=1.0).sum() / num_all, 0.5
#                     )
#                     punish += kill_ratio
#                 target = self.gamma
#                 loss_latency = torch.abs(budget_active / budget_all - target)
#                 loss_budget = loss_latency + punish
#                 tb_dict.update({'dbq_loss_latency': loss_latency.item(),
#                                 'dbq_loss_punish': punish.item(),
#                                 'budget_target': target})
#             else:
#                 raise NotImplementedError
#             self.shared_dict.clear()
#         return loss_budget, tb_dict
#
#     def compute_budget_map(func):
#         def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor,
#                     feats: torch.Tensor, new_feats: torch.Tensor,
#                     batch_id: torch.Tensor, new_batch_id: torch.Tensor):
#             b, n, m, k = xyz.shape[0], xyz.shape[1], new_xyz.shape[1], self.neighbour
#             self.title = f"{b} {n} {m} {k}"
#             prob = getattr(self, 'prob', 1.0)
#             self.mask = xyz.new_zeros([b, m])
#             active = torch.randperm(b * m)[:int(b * m * prob)]
#             self.mask.view(-1)[active] = 1
#             print(self.mask.sum().item())
#             with ScopeTimer(self.title + ": ", average=False) as t:
#                 ret = func(self, xyz, new_xyz, feats, new_feats, batch_id, new_batch_id)
#             self.duration = t.duration
#             return ret
#
#         return forward
#
#     def get_loss(self, tb_dict):
#         mask = self.shared_dict[self.id][0]
#
#         loss_budget, tb_dict1 = self.__try_get_loss_for_all_dbq()
#         loss_budget = self.weight * loss_budget
#
#         tb_dict.update({self.tag + '_ratio': mask.sum() / mask.new_tensor(mask.shape).prod().item(),
#                         'budget_loss': (loss_budget.item() if not isinstance(loss_budget, float) else loss_budget)})
#         tb_dict.update(tb_dict1)
#         return loss_budget, tb_dict
#
#     def assign_targets(self, batch_dict):
#         self.percentage = batch_dict['epoch_current'][0] / batch_dict['epoch_total'][0]
#
#     # @compute_budget_map
#     def forward(self,
#                 xyz: torch.Tensor, new_xyz: torch.Tensor,
#                 feats: torch.Tensor, new_feats: torch.Tensor,
#                 batch_id: torch.Tensor, new_batch_id: torch.Tensor):
#         """
#
#         Args:
#             xyz: [B, N, 3] source point coords
#             new_xyz: [B, M, 3] group center (queries) coords
#             feats: [B, C, N] features of source points
#             new_feats: [B, C, M] features of query points
#             batch_id: [B, N, 1]
#             new_batch_id: [B, M, 1]
#
#         Returns:
#             new_features: [B, C, M] features of each group (queries)
#             mask_float: [B, M]
#             mask: [B, M]
#         """
#         new_active_feats, mask_float, mask = self.__dbq(xyz, new_xyz, feats, new_feats, batch_id, new_batch_id)
#
#         if self.training:
#             self.id = len(self.shared_dict)
#             self.shared_dict[self.id] = [mask_float, self.a, self.b]
#
#         if new_active_feats is not None:
#             new_active_feats = new_active_feats.permute(2, 1, 0).contiguous()  # (K, C1, active)
#             new_active_feats = self.mlps(new_active_feats)  # (K, C2, active)
#             new_active_feats = torch.max(new_active_feats, dim=0)[0]
#         # this may case error "Expected to have finished reduction in the prior iteration before starting a new one"
#         # when distribute training
#         return [new_active_feats, mask_float, mask]
