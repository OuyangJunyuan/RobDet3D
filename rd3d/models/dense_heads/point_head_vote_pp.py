import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ..backbones_3d.pfe.point_set_abstraction import PointVoteInsCenter
from ..backbones_3d.pfe.ops.builder import sampler
from ...utils import box_coder_utils, box_utils, common_utils, loss_utils
from .point_head_template import PointHeadTemplate


class PointHeadVotePlusPlus(PointHeadTemplate):
    # TODO: IA-SSD no use init_weights and shared_layers.
    # TODO: IA-SSD use box iou3d layers.
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        self.predict_boxes_when_training = predict_boxes_when_training

        self.vote_sampler = sampler.from_cfg(self.model_cfg.VOTE_SAMPLER, input_channels=input_channels)
        self.vote_layers = nn.ModuleList()
        for vote_cfg in self.model_cfg.VOTE_MODULE:
            vote_cfg.input_channels = input_channels
            self.vote_layers.append(PointVoteInsCenter(vote_cfg))
            input_channels = self.vote_layers[-1].output_channels

        box_coder_cfg = self.model_cfg.BOX_CODER
        self.box_coder = getattr(box_coder_utils, box_coder_cfg.name)(**box_coder_cfg)

        self.shared_layers = self.make_fc_layers(
            fc_list=[input_channels] + self.model_cfg.SHARED_FC
        )
        input_channels = self.model_cfg.SHARED_FC[-1] if len(self.model_cfg.SHARED_FC) > 0 else input_channels

        self.cls_layers = self.make_fc_layers(
            fc_list=[input_channels] + self.model_cfg.CLS_FC,
            output_channels=num_class,
        )
        self.reg_layers = self.make_fc_layers(
            fc_list=[input_channels] + self.model_cfg.REG_FC,
            output_channels=self.box_coder.code_size,
        )
        self.iou_layers = self.make_fc_layers(
            fc_list=[input_channels] + self.model_cfg.IOU_FC,
            output_channels=1,
        )
        self.train_dict = {}

        self.init_weights(weight_init='xavier')

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        pred_scores, pred_classes = point_cls_preds.max(dim=-1)
        pred_classes += 1
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes)

        return point_box_preds, pred_scores, pred_classes

    ####################################################################################################################
    def make_fc_layers(self, fc_list, output_channels=None):
        fc_layers = [[nn.Linear(fc_list[k], fc_list[k + 1], bias=False),
                      nn.BatchNorm1d(fc_list[k + 1]),
                      nn.ReLU()] for k in range(fc_list.__len__() - 1)]
        if output_channels is not None:
            fc_layers.append([nn.Linear(fc_list[-1], output_channels, bias=True)])
        return nn.Sequential(*[ii for i in fc_layers for ii in i])

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        #
        # for m in self.modules():
        #     if m is not self and hasattr(m, 'init_weights'):
        #         m.init_weights()

    def build_losses(self, losses_cfg):
        # classification loss
        self.add_module('cls_loss_func', getattr(loss_utils, losses_cfg.LOSS_CLS)())

        # regression loss
        self.add_module('reg_loss_func', getattr(loss_utils, losses_cfg.LOSS_REG)())

    ####################################################################################################################
    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """

        def get_targets(gt_central_radius=2.0, extra_width=None, **kwargs):
            points = self.train_dict['point_vote_coords']
            gt_boxes = input_dict['gt_boxes']
            batch_size, num_points, _ = points.size()

            point_cls_labels = gt_boxes.new_zeros((batch_size, num_points), dtype=torch.long)
            point_reg_labels = gt_boxes.new_zeros((batch_size, num_points, self.box_coder.code_size))
            point_box_labels = gt_boxes.new_zeros((batch_size, num_points, gt_boxes.size(2) - 1))
            for k in range(batch_size):
                cur_points = points[k]
                cur_gt = gt_boxes[k][gt_boxes[k].sum(dim=-1) != 0]
                cur_point_cls_labels = point_cls_labels[k]

                # point in box checking
                box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    cur_points[None, ...].contiguous(),
                    cur_gt[None, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idx_of_pts >= 0)

                if gt_central_radius:
                    # set cls_labels to be ignored
                    cur_points_box_centers = cur_gt[box_idx_of_pts][..., :3].clone()
                    # indicate which points are not far away from gt_boxes
                    ball_flag = ((cur_points_box_centers - cur_points).norm(dim=-1) < gt_central_radius)
                    fg_flag = box_fg_flag & ball_flag  # the points in boxes and balls are labeled positive
                    ignore_flag = fg_flag ^ box_fg_flag  # ignore that points between boxes and balls.
                    cur_point_cls_labels[ignore_flag] = -1
                else:
                    cur_extend_gt = box_utils.enlarge_box3d(cur_gt, extra_width)
                    extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        cur_points[None, ...].contiguous(),
                        cur_extend_gt[None, :, 0:7].contiguous()
                    ).squeeze(dim=0)
                    fg_flag = box_fg_flag
                    ignore_flag = fg_flag ^ (extend_box_idx_of_pts >= 0)
                    cur_point_cls_labels[ignore_flag] = -1

                # set gt_boxes belong to each fg_points
                gt_box_of_fg_points = cur_gt[box_idx_of_pts[fg_flag]]

                # set cls_labels to be positive
                cur_point_cls_labels[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
                point_cls_labels[k] = cur_point_cls_labels

                # set reg_labels for fg_points via box_coder.encode
                if gt_box_of_fg_points.shape[0] > 0:
                    cur_point_reg_labels = point_reg_labels[k]
                    reg_labels_of_fg_points = self.box_coder.encode_torch(
                        points=cur_points[fg_flag],
                        gt_boxes=gt_box_of_fg_points[:, :-1],
                        gt_classes=gt_box_of_fg_points[:, -1].long()
                    )
                    cur_point_reg_labels[fg_flag] = reg_labels_of_fg_points
                    point_reg_labels[k] = cur_point_reg_labels

                    # set gt_boxes for each fg_points
                    cur_point_box_labels = point_box_labels[k]
                    cur_point_box_labels[fg_flag] = gt_box_of_fg_points[:, :-1]
                    point_box_labels[k] = cur_point_box_labels

            targets_dict = {
                'point_cls_labels': point_cls_labels.view(-1),
                'point_reg_labels': point_reg_labels.view(-1, point_reg_labels.shape[-1]),
                'point_box_labels': point_box_labels.view(-1, point_box_labels.shape[-1])
            }
            return targets_dict

        # handle target assigment in submodule
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    getattr(m, 'assign_targets', lambda x: 0)(input_dict)
            getattr(module, 'assign_targets', lambda x: 0)(input_dict)

        # handle target assigment in this module
        if self.model_cfg.TARGET_CONFIG.method == 'mask':
            self.train_dict.update(get_targets(**self.model_cfg.TARGET_CONFIG))
        else:
            raise NotImplementedError

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        def get_cls_layer_loss():
            @torch.no_grad()
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

            point_cls_labels = self.train_dict['point_cls_labels'].view(-1)
            point_cls_preds = self.train_dict['point_cls_preds'].view(-1, self.num_class)

            positives = point_cls_labels > 0
            negatives = point_cls_labels == 0
            cls_weights = positives * 1.0 + negatives * 1.0

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(value=1.0, dim=-1,
                                     index=(point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long())

            # class target multipy with centerness
            centerness_label = generate_centerness_label(self.train_dict['point_vote_coords'].view(-1, 3),
                                                         self.train_dict['point_box_labels'][..., :7].view(-1, 7),
                                                         positives)
            centerness_min = 0.0
            centerness_max = 1.0
            centerness_label = centerness_min + (centerness_max - centerness_min) * centerness_label
            one_hot_targets *= centerness_label.unsqueeze(dim=-1)

            # loss calculation
            point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets[..., 1:], weights=cls_weights)
            point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.WEIGHTS.point_cls_weight

            tb_dict.update({'point_pos_num': positives.sum().item()})
            return point_loss_cls, cls_weights, tb_dict  # point_loss_cls: (N)

        def get_box_layer_loss():
            def get_axis_aligned_iou_loss_lidar(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
                """
                Args:
                    pred_boxes: (N, 7) float Tensor.
                    gt_boxes: (N, 7) float Tensor.
                Returns:
                    iou_loss: (N) float Tensor.
                """
                assert pred_boxes.shape[0] == gt_boxes.shape[0]

                pos_p, len_p, *cps = torch.split(pred_boxes, 3, dim=-1)
                pos_g, len_g, *cgs = torch.split(gt_boxes, 3, dim=-1)

                len_p = torch.clamp(len_p, min=1e-5)
                len_g = torch.clamp(len_g, min=1e-5)
                vol_p = len_p.prod(dim=-1)
                vol_g = len_g.prod(dim=-1)

                min_p, max_p = pos_p - len_p / 2, pos_p + len_p / 2
                min_g, max_g = pos_g - len_g / 2, pos_g + len_g / 2

                min_max = torch.min(max_p, max_g)
                max_min = torch.max(min_p, min_g)
                diff = torch.clamp(min_max - max_min, min=0)
                intersection = diff.prod(dim=-1)
                union = vol_p + vol_g - intersection
                iou_axis_aligned = intersection / torch.clamp(union, min=1e-5)

                iou_loss = 1 - iou_axis_aligned
                return iou_loss

            def get_corner_loss_lidar(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
                """
                Args:
                    pred_boxes: (N, 7) float Tensor.
                    gt_boxes: (N, 7) float Tensor.
                Returns:
                    corner_loss: (N) float Tensor.
                """
                assert pred_boxes.shape[0] == gt_boxes.shape[0]

                pred_box_corners = box_utils.boxes_to_corners_3d(pred_boxes)
                gt_box_corners = box_utils.boxes_to_corners_3d(gt_boxes)

                gt_boxes_flip = gt_boxes.clone()
                gt_boxes_flip[:, 6] += np.pi
                gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_boxes_flip)
                # (N, 8, 3)
                corner_loss = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(pred_box_corners - gt_box_corners, 1.0)
                corner_loss_flip = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(
                    pred_box_corners - gt_box_corners_flip, 1.0)
                corner_loss = torch.min(corner_loss.sum(dim=2), corner_loss_flip.sum(dim=2))

                return corner_loss.mean(dim=1)

            pos_mask = self.train_dict['point_cls_labels'].view(-1) > 0
            point_reg_preds = self.train_dict['point_reg_preds']
            point_reg_labels = self.train_dict['point_reg_labels'].view(point_reg_preds.shape[0], -1)
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.WEIGHTS

            reg_weights = pos_mask.float()

            # calculate box regression loss
            point_loss_offset_reg = self.reg_loss_func(
                point_reg_preds[None, :, :6],
                point_reg_labels[None, :, :6],
                weights=reg_weights[None, ...]
            )
            point_loss_offset_reg = point_loss_offset_reg.sum(dim=-1).squeeze()

            if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
                point_loss_velo_reg = self.reg_loss_func(
                    point_reg_preds[None, :, 6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                    point_reg_labels[None, :,
                    6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                    weights=reg_weights[None, ...]
                )
                point_loss_velo_reg = point_loss_velo_reg.sum(dim=-1).squeeze()
                point_loss_offset_reg = point_loss_offset_reg + point_loss_velo_reg

            point_loss_offset_reg *= loss_weights_dict['point_offset_reg_weight']

            if isinstance(self.box_coder, box_coder_utils.PointBinResidualCoder):
                point_angle_cls_labels = \
                    point_reg_labels[:, 6:6 + self.box_coder.angle_bin_num]
                point_loss_angle_cls = F.cross_entropy(  # angle bin cls
                    point_reg_preds[:, 6:6 + self.box_coder.angle_bin_num],
                    point_angle_cls_labels.argmax(dim=-1), reduction='none') * reg_weights

                point_angle_reg_preds = point_reg_preds[:,
                                        6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
                point_angle_reg_labels = point_reg_labels[:,
                                         6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
                point_angle_reg_preds = (point_angle_reg_preds * point_angle_cls_labels).sum(dim=-1, keepdim=True)
                point_angle_reg_labels = (point_angle_reg_labels * point_angle_cls_labels).sum(dim=-1, keepdim=True)
                point_loss_angle_reg = self.reg_loss_func(
                    point_angle_reg_preds[None, ...],
                    point_angle_reg_labels[None, ...],
                    weights=reg_weights[None, ...]
                )
                point_loss_angle_reg = point_loss_angle_reg.squeeze()

                point_loss_angle_cls *= loss_weights_dict['point_angle_cls_weight']
                point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']

                point_loss_box = point_loss_offset_reg + point_loss_angle_cls + point_loss_angle_reg  # (N)
            else:
                point_angle_reg_preds = point_reg_preds[:, 6:]
                point_angle_reg_labels = point_reg_labels[:, 6:]
                point_loss_angle_reg = self.reg_loss_func(
                    point_angle_reg_preds[None, ...],
                    point_angle_reg_labels[None, ...],
                    weights=reg_weights[None, ...]
                )
                point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']
                point_loss_box = point_loss_offset_reg + point_loss_angle_reg

            if reg_weights.sum() > 0:
                point_box_preds = self.train_dict['point_box_preds']
                point_box_labels = self.train_dict['point_box_labels'].view(*point_box_preds.size())
                point_loss_box_aux = 0

                if self.model_cfg.LOSS_CONFIG.get('AXIS_ALIGNED_IOU_LOSS_REGULARIZATION', False):
                    point_loss_iou = get_axis_aligned_iou_loss_lidar(
                        point_box_preds[pos_mask, 0:7],
                        point_box_labels[pos_mask, 0:7]
                    )
                    point_loss_iou *= loss_weights_dict['point_iou_weight']
                    point_loss_box_aux = point_loss_box_aux + point_loss_iou

                if self.model_cfg.LOSS_CONFIG.get('CORNER_LOSS_REGULARIZATION', False):
                    point_loss_corner = get_corner_loss_lidar(
                        point_box_preds[pos_mask, 0:7],
                        point_box_labels[pos_mask, 0:7]
                    )
                    point_loss_corner *= loss_weights_dict['point_corner_weight']
                    point_loss_box_aux = point_loss_box_aux + point_loss_corner

                if True:
                    iou_pred = self.train_dict['point_iou_preds'][pos_mask] * 2 - 1
                    iou_target = (1 - get_axis_aligned_iou_loss_lidar(
                        point_box_preds[pos_mask, 0:7],
                        point_box_labels[pos_mask, 0:7]
                    )) * 2 - 1
                    point_loss_iou_reg = self.reg_loss_func(iou_pred[None, ..., 0], iou_target[None, ...])
                    point_loss_iou_reg *= loss_weights_dict['point_iou_reg_weight']
                point_loss_box[pos_mask] = point_loss_box[pos_mask] + point_loss_box_aux + point_loss_iou_reg

            return point_loss_box, reg_weights, tb_dict  # point_loss_box: (N)

        def get_submodule_loss(sm):
            if hasattr(sm, 'get_loss'):
                loss, tb_dict_ = sm.get_loss(tb_dict)
                tb_dict.update(tb_dict_)
                return loss
            return 0

        point_loss_cls, cls_weights, tb_dict_1 = get_cls_layer_loss()
        point_loss_box, box_weights, tb_dict_2 = get_box_layer_loss()
        point_loss_cls = point_loss_cls.sum() / torch.clamp(cls_weights.sum(), min=1.0)
        point_loss_box = point_loss_box.sum() / torch.clamp(box_weights.sum(), min=1.0)
        tb_dict.update({
            'point_cls_loss': point_loss_cls.item(),
            'point_box_loss': point_loss_box.item()
        })
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        point_loss = point_loss_cls + point_loss_box

        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    point_loss = point_loss + get_submodule_loss(m)
            point_loss = point_loss + get_submodule_loss(module)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_batch_id:             (B, N, 1)
                point_coords:               (B, N, 3)
                point_features:             (B, N, Ci)
                ------------------------------
                point_scores (optional):    (B, N)
                gt_boxes (optional):        (B, M, 8)
        Returns:
            batch_dict:
                # point_cls_scores: (N1 + N2 + N3 + ..., 1)
                # point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        # input data assignment
        batch_size = batch_dict['batch_size']
        point_coords = batch_dict['point_coords']  # (B,N,3)
        point_features = batch_dict['point_features']  # (B,N,Ci)
        point_batch_id = batch_dict['point_batch_id']  # (B,N,1)

        # select which points to vote
        vote_ind = self.vote_sampler(point_coords, point_features)
        center_coords = common_utils.gather(point_coords, vote_ind)  # (B,M,3)
        center_feats = common_utils.gather(point_features, vote_ind)  # (B,M,Ci)
        center_bid = point_batch_id[:, :vote_ind.shape[1], :] if point_batch_id is not None else None  # (B,M,1)

        # generate vote points
        for vote_layer in self.vote_layers:
            center_coords, center_feats = vote_layer(center_coords, center_feats, center_bid,
                                                     point_coords, point_features, point_batch_id)  # (B,M,3) (B,M,C1)

        shared_feats = self.shared_layers(center_feats.view(-1, center_feats.shape[-1]))  # (B*M,C2)
        point_cls_preds = self.cls_layers(shared_feats)
        point_reg_preds = self.reg_layers(shared_feats)
        point_iou_preds = self.iou_layers(shared_feats)

        point_box_preds, point_score_preds, point_label_preds = self.generate_predicted_boxes(
            points=center_coords.view(-1, 3),
            point_cls_preds=point_cls_preds,
            point_box_preds=point_reg_preds
        )

        point_iou_preds = (point_iou_preds + 1) / 2
        if self.training:
            self.train_dict.update({'point_vote_coords': center_coords,
                                    'point_cls_preds': point_cls_preds,
                                    'point_reg_preds': point_reg_preds,
                                    'point_box_preds': point_box_preds,
                                    'point_iou_preds': point_iou_preds})

        if not self.training or self.predict_boxes_when_training:
            batch_dict['batch_cls_preds'] = point_cls_preds.view(batch_size, -1, point_cls_preds.shape[-1])
            point_iou_preds = point_iou_preds.view(batch_size, -1, 1)
            point_box_preds = point_box_preds.view(batch_size, -1, point_box_preds.shape[-1])

            if not torch.onnx.is_in_onnx_export():
                # the labels and scores will be calculated again in postprocessing for normal evaluation.
                batch_dict['batch_box_preds'] = torch.concat([point_box_preds, point_iou_preds], dim=-1)
                batch_dict['cls_preds_normalized'] = False
            else:
                # the labels and scores are saved to avoid calling max() again under export mode.
                point_label_preds = point_label_preds.view(batch_size, -1, 1).float()
                batch_dict['batch_score_preds'] = point_score_preds.view(batch_size, -1, 1)
                batch_dict['batch_box_preds'] = torch.cat([point_box_preds, point_iou_preds, point_label_preds], dim=-1)
        return batch_dict
