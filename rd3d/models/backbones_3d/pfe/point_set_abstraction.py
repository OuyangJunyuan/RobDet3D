import numpy as np
import torch
import torch.nn as nn
from ....utils.common_utils import gather, ScopeTimer, apply1d
from .ops.builder import sampler, grouper, aggregation


class GeneralPointSetAbstraction(nn.Module):
    """
    stack of multi samplers and groupers
    """

    def __init__(self, ab_layer_cfg, input_channels: int):
        super(GeneralPointSetAbstraction, self).__init__()
        self.cfg = ab_layer_cfg
        self.input_channels = self.cfg.get("input_channels", input_channels)

        self.samplers = nn.ModuleList()
        for sampler_cfg in self.cfg.get('samplers', []):
            self.samplers.append(sampler.from_cfg(sampler_cfg, input_channels=self.input_channels))
        self.sampler_output_indices = sum([s.indices_as_output for s in self.samplers]) != 0

        self.groupers = nn.ModuleList()
        for grouper_cfg in self.cfg.get('groupers', []):
            self.groupers.append(grouper.from_cfg(grouper_cfg, input_channels=self.input_channels, ctx=self.samplers))

        self.agg = aggregation.from_cfg(self.cfg.aggregation, input_channels=[g.output_channels for g in self.groupers])

        self.output_channels = self.agg.output_channels
        self.need_new_feats = True in [grouping.need_query_features for grouping in self.groupers]

    def assign_targets(self, batch_dict):
        # handle target assigment in submodule
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    getattr(m, 'assign_targets', lambda x: 0)(batch_dict)
            getattr(module, 'assign_targets', lambda x: 0)(batch_dict)

    def get_loss(self, tb_dict):
        def get_submodule_loss(sm):
            if hasattr(sm, 'get_loss'):
                loss, tb_dict_ = sm.get_loss(tb_dict)
                tb_dict.update(tb_dict_)
                return loss
            return 0

        sa_loss = 0
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    sa_loss = sa_loss + get_submodule_loss(m)
            sa_loss = sa_loss + get_submodule_loss(module)
        return sa_loss, tb_dict

    def forward(self,
                xyz: torch.Tensor, feats: torch.Tensor = None, bid=None,
                new_xyz=None, new_feats=None, new_bid=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)
            feats: (B, N, Ci)
            bid: (B, N, 1)
            new_xyz: (B, M, 3)
            new_feats: (B, M, Ci)
            new_bid: (B, M, 1)

        Returns:
            new_xyz: (B, M, 3)
            new_feats: (B, M, Co)
        """
        if new_xyz is None:
            if self.sampler_output_indices:
                sample_ind = [s(xyz=xyz, feats=feats, bid=bid) for s in self.samplers]  # (B,M)
                sample_ind = sample_ind[0] if len(sample_ind) == 1 else torch.cat(sample_ind, dim=-1)
                new_xyz = gather(xyz, sample_ind)  # (B,M,3)
                new_feats = gather(feats, sample_ind) if self.need_new_feats and not new_feats else new_feats  # (B,M,C)
            else:
                new_xyz = self.samplers[0](xyz=xyz, feats=feats)

        new_bid = new_bid if bid is None else bid[:, :new_xyz.shape[1], :]  # (B,M,1)
        group_list = [g(xyz, new_xyz, feats, new_feats, bid, new_bid) for g in self.groupers]
        new_feats = self.agg(*[list(ret_i) for ret_i in zip(*group_list)])
        return new_xyz, new_feats, new_bid


class PointVoteInsCenter(nn.Module):
    """
        {
            mlps: [128], max_translation_range: [3.0, 3.0, 2.0],
            sa: {
                    groupers: [ {name: 'ball', query: {radius: 4.8, neighbour: 48}, mlps: [256, 256, 512]},
                                {name: 'ball', query: {radius: 6.4, neighbour: 64}, mlps: [256, 256, 1024]}]
                    aggregation: {pooling: 'max_pool'}
                }
         }
    """

    def __init__(self, module_cfg):
        super(PointVoteInsCenter, self).__init__()
        self.cfg = module_cfg
        self.input_channels = self.cfg.input_channels
        self.max_translation_range = self.cfg.max_translation_range

        mlp_cfg = [self.input_channels] + self.cfg.mlps
        fc_layers = [[nn.Linear(mlp_cfg[k], mlp_cfg[k + 1], bias=False),
                      nn.BatchNorm1d(mlp_cfg[k + 1]),
                      nn.ReLU()] for k in range(len(mlp_cfg) - 1)]
        fc_layers.append([nn.Linear(mlp_cfg[-1], 3, bias=True)])
        self.vote_layers = nn.Sequential(*[ii for i in fc_layers for ii in i])
        if 'sa' in module_cfg:
            self.sa_layers = GeneralPointSetAbstraction(module_cfg.sa, self.input_channels)
            self.output_channels = self.sa_layers.output_channels
        else:
            self.sa_layers = None
            self.output_channels = self.input_channels
        self.train_dict = {}
        from ....utils.loss_utils import WeightedSmoothL1Loss
        self.reg_loss_func = WeightedSmoothL1Loss()

    def assign_targets(self, batch_dict):
        from ....ops.roiaware_pool3d import roiaware_pool3d_utils
        from ....utils import box_utils

        def get_point_targets(points, gt_boxes, set_ignore_flag=False, extra_width=None):
            """
            only batch point were supported.
            """
            batch_size, num_pts, _ = points.size()
            point_cls_labels = points.new_zeros((batch_size, num_pts), dtype=torch.long)
            point_reg_labels = gt_boxes.new_zeros((batch_size, num_pts, 3))
            for k in range(batch_size):
                cur_points = points[k]
                cur_gt = gt_boxes[k][gt_boxes[k].sum(dim=-1) != 0]
                cur_extend_gt = cur_gt if extra_width is None else box_utils.enlarge_box3d(cur_gt, extra_width)
                cur_point_cls_labels = point_cls_labels[k]
                cur_point_reg_labels = point_reg_labels[k]

                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    cur_points[None, ...].contiguous(),
                    cur_extend_gt[None, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                if set_ignore_flag:
                    box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        cur_points[None, ...].contiguous(),
                        cur_gt[None, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    box_fg_flag = (box_idx_of_pts >= 0)
                    ignore_flag = box_fg_flag ^ (extend_box_idx_of_pts >= 0)
                    cur_point_cls_labels[ignore_flag] = -1
                else:
                    box_fg_flag = (extend_box_idx_of_pts >= 0)
                    box_idx_of_pts = extend_box_idx_of_pts

                cur_point_cls_labels[box_fg_flag] = 1
                point_cls_labels[k] = cur_point_cls_labels

                cur_point_reg_labels[box_fg_flag] = cur_gt[box_idx_of_pts[box_fg_flag]][:, :3]
                point_reg_labels[k] = cur_point_reg_labels
            return point_cls_labels, point_reg_labels

        # handle target assigment in submodule
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    getattr(m, 'assign_targets', lambda x: 0)(batch_dict)
            getattr(module, 'assign_targets', lambda x: 0)(batch_dict)

        # handle target assigment in this module
        point_cls_labels, point_reg_labels = get_point_targets(self.train_dict['coords'], batch_dict['gt_boxes'],
                                                               **(self.cfg.train.get('target', {})))

        self.train_dict.update({'vote_cls_labels': point_cls_labels, 'vote_reg_labels': point_reg_labels})

    def get_loss(self, tb_dict=None):
        def get_loss_this_module():
            pos_mask = self.train_dict['vote_cls_labels'].view(-1) > 0
            vote_reg_labels = self.train_dict['vote_reg_labels'].view(-1, 3)
            vote_reg_preds = self.train_dict['vote_reg_preds'].view(-1, 3)

            reg_weights = pos_mask.float()
            reg_weights /= torch.clamp(pos_mask.sum().float(), min=1.0)

            vote_loss_reg_src = self.reg_loss_func(
                vote_reg_preds[None, ...],
                vote_reg_labels[None, ...],
                weights=reg_weights[None, ...])

            vote_loss_reg = vote_loss_reg_src.sum() * self.cfg.train.loss.weight

            tb_dict.update({self.cfg.train.loss.tb_tag: vote_loss_reg.item()})
            return vote_loss_reg, tb_dict

        def get_submodule_loss(sm):
            if hasattr(sm, 'get_loss'):
                loss, tb_dict_ = sm.get_loss(tb_dict)
                tb_dict.update(tb_dict_)
                return loss
            return 0

        tb_dict = {} if tb_dict is None else tb_dict

        loss, tb_dict = get_loss_this_module()

        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    loss += get_submodule_loss(m)
            loss += get_submodule_loss(module)
        self.train_dict.clear()
        return loss, tb_dict

    def forward(self,
                start_xyz: torch.Tensor, start_feats: torch.Tensor, start_bid: torch.Tensor,
                sa_xyz: torch.Tensor, sa_feats: torch.Tensor, sa_bid: torch.Tensor):
        """
        Args:
            start_xyz: where the center voting from                     (B, M, 3)
            start_feats: the feats to predict the offset for start_xyz  (B, M, Ci_1)
            start_bid:
            sa_xyz: the source xyz to be grouped in SA module           (B, N, 3)
            sa_feats: the feats of sa_xyz                               (B, N, Ci_2)
            sa_bid:
        Returns:
            vote_end: the coords voted from start_xyz                   (B, M, 3)
            vote_end_feats: the feats of vote_end                       (B, M, Co)
        """
        vote_offsets = apply1d(self.vote_layers, start_feats)

        max_translation = start_xyz.new_tensor(self.max_translation_range)[None, None, ...]
        vote_offsets = torch.max(vote_offsets, -max_translation)
        vote_offsets = torch.min(vote_offsets, max_translation)
        vote_end = vote_offsets + start_xyz

        _, vote_end_feats, _ = self.sa_layers(
            new_xyz=vote_end,
            new_feats=start_feats,
            new_bid=start_bid,
            xyz=sa_xyz,
            feats=sa_feats,
            bid=sa_bid
        ) if self.sa_layers is not None else (None, start_feats, None)

        if self.training:
            self.train_dict.update({'coords': start_xyz, 'vote_reg_preds': vote_end})
        return vote_end, vote_end_feats
