import time
import torch
import torch.nn as nn
from .builder import sampler


class PointSampling(nn.Module):
    def __init__(self, sampling_cfg):
        super(PointSampling, self).__init__()
        self.cfg = sampling_cfg
        self.range = self.cfg.get('range', None)
        self.num_sample = self.cfg.sample
        self.output_indices = True

    @staticmethod
    def sample_in_range(func):
        def wrapper(self, xyz: torch.Tensor, feats: torch.Tensor = None, bid: torch.Tensor = None, **kwargs):
            if self.range is None or (self.range[0] == 0 and self.range[1] == xyz.shape[1]):
                return func(self, xyz=xyz, feats=feats, bid=bid, **kwargs)
            else:
                bid_in_range = bid[:, self.range[0]:self.range[1], ...].contiguous() if bid is not None else bid
                xyz_in_range = xyz[:, self.range[0]:self.range[1], ...].contiguous() if xyz is not None else xyz
                feats_in_range = feats[:, self.range[0]:self.range[1], ...].contiguous() if feats is not None else feats
                return func(self, xyz=xyz_in_range, feats=feats_in_range, bid=bid_in_range, **kwargs) + self.range[0]

        return wrapper

    @staticmethod
    def build_mlps(mlp_cfg, in_channels, out_channels=1):
        shared_mlp = []
        for k in range(len(mlp_cfg)):
            shared_mlp.extend([
                nn.Linear(in_channels, mlp_cfg[k], bias=False),
                nn.BatchNorm1d(mlp_cfg[k]),
                nn.ReLU()
            ])
            in_channels = mlp_cfg[k]
        shared_mlp.append(
            nn.Linear(in_channels, out_channels, bias=True),
        )
        return nn.Sequential(*shared_mlp)


PS = PointSampling


@sampler.register_module('select', 'index')
class IndexPointSampling(PS):
    def __init__(self, sampling_cfg, *args, **kwargs):
        super(IndexPointSampling, self).__init__(sampling_cfg)

    @torch.no_grad()
    @PS.sample_in_range
    def forward(self, xyz: torch.Tensor, **kwargs) -> torch.Tensor:
        indices = torch.arange(self.num_sample[0], self.num_sample[1], dtype=torch.long, device=xyz.device)
        return indices.expand(xyz.shape[0], -1).contiguous()


@sampler.register_module('rps', 'rand')
class RandomPointSampling(PS):
    def __init__(self, sampling_cfg, *args, **kwargs):
        super(RandomPointSampling, self).__init__(sampling_cfg)

    @torch.no_grad()
    @PS.sample_in_range
    def forward(self, xyz: torch.Tensor, **kwargs) -> torch.Tensor:
        indices = torch.vstack(
            [
                torch.randperm(xyz.shape[1], dtype=torch.long, device=xyz.device)[:self.num_sample]
                for _ in range(xyz.shape[0])
            ]
        )
        return indices


@sampler.register_module('rvs')
class RandomVoxelSampling(PS):
    def __init__(self, sampling_cfg, *args, **kwargs):
        super(RandomVoxelSampling, self).__init__(sampling_cfg)
        self.output_indices = False
        self.voxel = sampling_cfg.voxel
        self.coors_range = sampling_cfg.coors_range
        self.channel = sampling_cfg.get('channel', 3)
        self.pool = sampling_cfg.get('pool', 'rand')
        self.max_pts_per_voxel = 1 if self.pool == 'rand' else sampling_cfg.get('max_pts_per_voxel', 5)
        self.adaptive = sampling_cfg.get('adaptive', False)
        self.gen = None

    def one_sample(self, gen, xyz, *args, **kwargs):
        voxels, _, num_pts_per_voxel, _ = gen.generate_voxel_with_id(xyz)
        if self.pool == 'rand':
            sampled_xyz = voxels[:, 0, :]
        elif self.pool == 'mean':
            points_mean = voxels.sum(dim=1, keepdim=False)  # (n,1,c)
            normalizer = torch.clamp_min(num_pts_per_voxel.view(-1, 1), min=1.0).type_as(points_mean)
            sampled_xyz = points_mean / normalizer
        else:
            raise NotImplementedError

        sampled_num = len(sampled_xyz)
        resample_num = self.num_sample - sampled_num
        if resample_num > 0:
            indices = torch.concat([torch.arange(0, sampled_num), torch.randint(sampled_num, (resample_num,))])
            sampled_xyz = sampled_xyz[indices]
        return sampled_xyz.unsqueeze(dim=0)

    def build_voxel_gen(self, voxel, device):
        from spconv.pytorch.utils import PointToVoxel
        return PointToVoxel(vsize_xyz=voxel,
                            coors_range_xyz=self.coors_range,
                            max_num_voxels=self.num_sample,
                            max_num_points_per_voxel=self.max_pts_per_voxel,
                            num_point_features=self.channel,
                            device=device)

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.adaptive:
            self.gen: HierarchicalAdaptiveVoxelSampling = sampler.from_cfg(
                dict(name='havs', sample=self.num_sample, voxel=self.voxel, return_detail=True)
            )
            self.gen(xyz)
            voxels = self.gen.return_dict["voxel_sizes"][:, 3:6].tolist()
            sampled_xyz = torch.cat([self.one_sample(self.build_voxel_gen(voxels[i], xyz.device), xyz[i])
                                     for i in range(xyz.shape[0])], dim=0)
        else:
            if self.gen is None:
                self.gen = self.build_voxel_gen(self.voxel, xyz.device)
            sampled_xyz = torch.cat([self.one_sample(self.gen, xyz[i]) for i in range(xyz.shape[0])], dim=0)

        return sampled_xyz


@sampler.register_module('fps', 'd-fps')
class FarthestPointSampling(PS):
    def __init__(self, sampling_cfg, *args, **kwargs):
        super(FarthestPointSampling, self).__init__(sampling_cfg)

    @torch.no_grad()
    @PS.sample_in_range
    def forward(self, xyz: torch.Tensor, **kwargs) -> torch.Tensor:
        from .....ops.pointnet2.pointnet2_batch import FPS
        indices = FPS.apply(xyz, self.num_sample)
        return indices.long()


@sampler.register_module('f-fps')
class FeatureFarthestPointSampling(PS):
    def __init__(self, sampling_cfg, *args, **kwargs):
        super(FeatureFarthestPointSampling, self).__init__(sampling_cfg)
        self.weight_gamma = self.cfg.gamma

    @torch.no_grad()
    @PS.sample_in_range
    def forward(self, xyz: torch.Tensor, feats: torch.Tensor = None, **kwargs) -> torch.Tensor:
        from .....ops.pointnet2.pointnet2_batch import furthest_point_sample_matrix
        dist = torch.cdist(xyz, xyz)
        if feats is not None:
            dist += torch.cdist(feats, feats) * self.weight_gamma

        indices = furthest_point_sample_matrix(dist, self.num_sample)
        return indices.long()


@sampler.register_module('s-fps')
class SemanticFarthestPointSampling(PS):
    def __init__(self, sampling_cfg, input_channels, *args, **kwargs):
        from .....utils.loss_utils import WeightedBinaryCrossEntropyLoss

        super(SemanticFarthestPointSampling, self).__init__(sampling_cfg)

        self.weight_gamma = self.cfg.gamma
        self.mlps = self.build_mlps(self.cfg.mlps, in_channels=input_channels, out_channels=1)
        self.train_dict = {}
        self.loss_func = WeightedBinaryCrossEntropyLoss()

    def assign_targets(self, batch_dict):
        from .....ops.roiaware_pool3d import roiaware_pool3d_utils
        from .....utils import box_utils

        def get_point_label(points, gt_boxes, set_ignore_flag=True, extra_width=None):
            """

            Args:
                points: [b,n,3]
                gt_boxes: [b,m,7-8]
                extra_width:
                set_ignore_flag:

            Returns:

            """
            batch_size, num_points, _ = points.size()
            point_cls_labels = points.new_zeros([batch_size, num_points], dtype=torch.long)
            for k in range(batch_size):
                cur_points = points[k]
                cur_gt = gt_boxes[k][gt_boxes[k].sum(dim=-1) != 0]
                cur_extend_gt = cur_gt if extra_width is None else box_utils.enlarge_box3d(cur_gt, extra_width)
                cur_point_cls_labels = point_cls_labels[k]

                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    cur_points[None, ...].contiguous(),
                    cur_extend_gt[None, :, 0:7].contiguous()
                ).squeeze(dim=0)
                if set_ignore_flag:
                    box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        cur_points[None, ...].contiguous(),
                        cur_gt[None, :, 0:7].contiguous()
                    ).squeeze(dim=0)
                    box_fg_flag = (box_idx_of_pts >= 0)

                    ignore_flag = box_fg_flag ^ (extend_box_idx_of_pts >= 0)
                    cur_point_cls_labels[ignore_flag] = -1
                else:
                    box_fg_flag = (extend_box_idx_of_pts >= 0)

                cur_point_cls_labels[box_fg_flag] = 1
                point_cls_labels[k] = cur_point_cls_labels

            return point_cls_labels.view(-1)  # (N, ) 0: bg, 1: fg, -1: ignore

        labels = get_point_label(self.train_dict['coords'], batch_dict['gt_boxes'],
                                 **(self.cfg.train.get('target', {})))
        self.train_dict.update({'labels': labels})

    def get_loss(self, tb_dict):
        scores, labels, weight = self.train_dict['scores'], self.train_dict['labels'], self.cfg.train.loss.weight

        positives, negatives = labels > 0, labels == 0
        cls_weights = positives * 1.0 + negatives * 1.0  # (N, 1)
        pos_normalizer = cls_weights.sum(dim=0).float()

        one_hot_targets = scores.new_zeros(*list(labels.shape), 2)
        one_hot_targets.scatter_(-1, (labels > 0).long().unsqueeze(-1), 1.0)
        one_hot_targets = one_hot_targets[:, 1:]  # (N, 1)

        loss = self.loss_func(scores[None],
                              one_hot_targets[None],
                              cls_weights.reshape(1, -1))
        loss = weight * loss.sum() / torch.clamp(pos_normalizer, min=1.0)
        self.train_dict.clear()
        return loss, {self.cfg.train.loss.tb_tag: loss.item()}

    @PS.sample_in_range
    def forward(self, xyz: torch.Tensor, feats: torch.Tensor, **kwargs) -> torch.Tensor:
        from .....ops.pointnet2.pointnet2_batch import furthest_point_sample_weights
        b, n, c = feats.size()

        scores = self.mlps(feats.view(-1, c)).view(b, n)  # (B, N)
        weights = scores.sigmoid() ** self.weight_gamma
        indices = furthest_point_sample_weights(xyz, weights, self.num_sample)

        if self.training:
            self.train_dict.update({'coords': xyz, 'scores': scores.view(-1, 1)})
        return indices.long()


@sampler.register_module('ctr_aware', 'ctr')
class CenterAwareSampling(PS):
    def __init__(self, sampling_cfg, input_channels, *args, **kwargs):
        from .....utils.loss_utils import WeightedClassificationLoss

        super(CenterAwareSampling, self).__init__(sampling_cfg)
        self.output_channels = len(self.cfg.get('class_names'))
        self.mlps = self.build_mlps(self.cfg.mlps,
                                    in_channels=input_channels,
                                    out_channels=self.output_channels)
        self.train_dict = {}
        self.loss_func = WeightedClassificationLoss()

    def assign_targets(self, batch_dict):
        from .....ops.roiaware_pool3d import roiaware_pool3d_utils
        from .....utils import box_utils

        def get_point_label(points, gt_boxes, extra_width=None):
            batch_size, num_points, _ = points.size()
            point_cls_labels = points.new_zeros([batch_size, num_points], dtype=torch.long)
            point_box_labels = gt_boxes.new_zeros((batch_size, num_points, gt_boxes.size(2) - 1))
            for k in range(batch_size):
                cur_points = points[k]
                cur_gt = gt_boxes[k][gt_boxes[k].sum(dim=-1) != 0]
                cur_extend_gt = cur_gt if extra_width is None else box_utils.enlarge_box3d(cur_gt, extra_width)
                cur_point_cls_labels = point_cls_labels[k]

                box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    cur_points[None, ...].contiguous(),
                    cur_gt[None, :, 0:7].contiguous()
                ).squeeze(dim=0).long()
                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    cur_points[None, ...].contiguous(),
                    cur_extend_gt[None, :, 0:7].contiguous()
                ).squeeze(dim=0).long()

                box_fg_flag, extend_box_fg_flag = (box_idx_of_pts >= 0), (extend_box_idx_of_pts >= 0)
                # instance points should keep unchanged
                extend_box_idx_of_pts[box_fg_flag] = box_idx_of_pts[box_fg_flag]
                # use extended box to label
                box_fg_flag, box_idx_of_pts = extend_box_fg_flag, extend_box_idx_of_pts

                gt_of_fg = cur_gt[box_idx_of_pts[box_fg_flag]]

                cur_point_cls_labels[box_fg_flag] = 1 if self.output_channels == 1 else gt_of_fg[:, -1].long()
                point_cls_labels[k] = cur_point_cls_labels

                cur_point_box_labels = point_box_labels[k]
                cur_point_box_labels[box_fg_flag] = gt_of_fg[:, :-1]
                point_box_labels[k] = cur_point_box_labels

            targets_dict = {
                'point_cls_labels': point_cls_labels.view(-1),  # (N, ) 0: bg, 1: fg, -1: ignore
                'point_box_labels': point_box_labels.view(-1, point_box_labels.shape[-1])
            }
            return targets_dict

        t_dict = get_point_label(self.train_dict['point_coords'], batch_dict['gt_boxes'],
                                 **(self.cfg.train.get('target', {})))
        self.train_dict.update(t_dict)

    def get_loss(self, tb_dict):
        from .....utils import common_utils

        def generate_centerness_label(point_base, point_box_labels, pos_mask, epsilon=1e-6):
            """
            Args:
                point_base: (N1 + N2 + N3 + ..., 3)
                point_box_labels: (N1 + N2 + N3 + ..., 7)
                pos_mask: (N1 + N2 + N3 + ...)
                epsilon:
            Returns:
                centerness_label: (N1 + N2 + N3 + ...)
            """
            centerness = point_box_labels.new_zeros(pos_mask.shape)

            point_box_labels = point_box_labels[pos_mask, :]
            canonical_xyz = point_base[pos_mask, :] - point_box_labels[:, :3]
            rys = point_box_labels[:, -1]
            canonical_xyz = common_utils.rotate_points_along_z(
                canonical_xyz.unsqueeze(dim=1), -rys
            ).squeeze(dim=1)

            distance_front = point_box_labels[:, 3] / 2 - canonical_xyz[:, 0]
            distance_back = point_box_labels[:, 3] / 2 + canonical_xyz[:, 0]
            distance_left = point_box_labels[:, 4] / 2 - canonical_xyz[:, 1]
            distance_right = point_box_labels[:, 4] / 2 + canonical_xyz[:, 1]
            distance_top = point_box_labels[:, 5] / 2 - canonical_xyz[:, 2]
            distance_bottom = point_box_labels[:, 5] / 2 + canonical_xyz[:, 2]

            centerness_l = torch.min(distance_front, distance_back) / torch.max(distance_front, distance_back)
            centerness_w = torch.min(distance_left, distance_right) / torch.max(distance_left, distance_right)
            centerness_h = torch.min(distance_top, distance_bottom) / torch.max(distance_top, distance_bottom)
            centerness_pos = torch.clamp(centerness_l * centerness_w * centerness_h, min=epsilon) ** (1 / 3.0)

            centerness[pos_mask] = centerness_pos

            return centerness

        weight = self.cfg.train.loss.weight
        point_cls_labels = self.train_dict['point_cls_labels'].contiguous().view(-1)
        point_cls_preds = self.train_dict['point_cls_preds'].contiguous().view(-1, self.output_channels)

        positives, negatives = point_cls_labels > 0, point_cls_labels == 0
        cls_weights = positives * 1.0 + negatives * 1.0
        pos_normalizer = cls_weights.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.output_channels + 1)
        one_hot_targets.scatter_(value=1.0, dim=-1,
                                 index=(point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long())

        # class target multipy with centerness
        centerness_label = generate_centerness_label(self.train_dict['point_coords'].view(-1, 3),
                                                     self.train_dict['point_box_labels'][..., :7].view(-1, 7),
                                                     positives)

        centerness_min = 0.0
        centerness_max = 1.0
        centerness_label = centerness_min + (centerness_max - centerness_min) * centerness_label
        one_hot_targets *= centerness_label.unsqueeze(dim=-1)

        # loss calculation
        point_loss_cls = self.loss_func(point_cls_preds,
                                        one_hot_targets[..., 1:],
                                        weights=cls_weights
                                        ).mean(dim=-1).sum()

        point_loss_cls = point_loss_cls * weight
        self.train_dict.clear()
        return point_loss_cls, {self.cfg.train.loss.tb_tag: point_loss_cls.item()}

    @PS.sample_in_range
    def forward(self, xyz: torch.Tensor, feats: torch.Tensor, **kwargs) -> torch.Tensor:
        b, n, c = feats.shape
        weights = self.mlps(feats.view(-1, c)).view(b, n, -1)
        cls_features_max, _ = weights.max(dim=-1)
        score_pred = torch.sigmoid(cls_features_max)  # (b, n)
        _, sample_idx = torch.topk(score_pred, self.num_sample, dim=-1)
        if self.training:
            self.train_dict.update({'point_cls_preds': weights, 'point_coords': xyz})
        return sample_idx.long()


@sampler.register_module('hvcs_v2', 'hvcs', 'havs')
class HierarchicalAdaptiveVoxelSampling(PointSampling):

    def __init__(self, sampling_cfg, *args, **kwargs):
        super(HierarchicalAdaptiveVoxelSampling, self).__init__(sampling_cfg)
        self.voxel = sampling_cfg.voxel
        self.tolerance = sampling_cfg.get('tolerance', 0.01)
        self.max_iter = sampling_cfg.get('max_iterations', sampling_cfg.get('max_iter', 20))
        self.keep_details = sampling_cfg.get('return_detail', False)
        self.keep_hashes = sampling_cfg.get('return_hash', False)
        self.return_dict = {}

    @torch.no_grad()
    @PS.sample_in_range
    def forward(self, xyz: torch.Tensor, **kwargs) -> torch.Tensor:
        from .....ops.havs import havs_batch

        indices = havs_batch(
            xyz, self.num_sample,
            self.voxel, self.tolerance, self.max_iter,
            self.keep_details, self.keep_hashes
        )
        if self.keep_details:
            indices, *infos = indices
            self.return_dict.update(voxel_sizes=infos[0], num_voxels=infos[1], sampled_masks=infos[2])
        if self.keep_hashes:
            indices, *infos = indices
            self.return_dict.update(voxel_sizes=infos[0], hash_tables=infos[1], subset_tables=infos[2])
        return indices
