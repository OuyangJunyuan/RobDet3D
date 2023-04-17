import torch
import numpy as np

from typing import List
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from ...utils.base import Register, merge_dicts

AUGMENTS = Register('AUGMENTS')


class Augment:
    def __init__(self, kwargs):
        self.name = type(self).__name__
        self.prob = kwargs.get('prob', 1.0)

    def random(self, data_dict):
        return {}

    def forward(self, data_dict, rand):
        pass

    def backward(self, data_dict, rand):
        pass

    # 如无给定rand，则用所记录的logs
    def invert(self, data_dict, rand=None):
        rand = rand or data_dict['aug_logs'][self.name]
        rand = {k: np.array(v) for k, v in (merge_dicts(rand) if isinstance(rand, list) else rand).items()}
        self.backward(data_dict, EasyDict(rand))
        return data_dict

    # 如无给定rand，则随机生成
    def __call__(self, data_dict, rand=None):
        enable = np.random.choice([False, True], p=[1 - self.prob, self.prob], replace=False)
        if enable:
            rand = rand or self.random(data_dict)
            rand = {k: np.array(v) for k, v in (merge_dicts(rand) if isinstance(rand, list) else rand).items()}
            self.forward(data_dict, EasyDict(rand))
            if 'aug_logs' not in data_dict: data_dict['aug_logs'] = {}
            data_dict['aug_logs'][self.name] = rand
        return data_dict


@AUGMENTS.register_module('global_rotate')
class GlobalRotate(Augment):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [-np.pi / 4, np.pi / 4])
        assert len(self.range) == 2

    @staticmethod
    def rotate_boxes(boxes, rot):
        if boxes.shape[-1] > 7:
            if isinstance(boxes, torch.Tensor):
                velo_xy = torch.concat([boxes[..., 7:9], torch.zeros_like(boxes[..., :1])], dim=-1)
            else:
                velo_xy = np.concatenate([boxes[..., 7:9], np.zeros_like(boxes[..., :1])], axis=-1)
            boxes[..., 7:9] = GlobalRotate.rotate_points(velo_xy, rot)[..., :2]

        boxes[..., :3] = GlobalRotate.rotate_points(boxes[..., :3], rot)
        if isinstance(boxes, torch.Tensor):
            rot = boxes.new_tensor(rot)
        boxes[..., 6] = boxes[..., 6] + rot[..., None]

        return boxes

    @staticmethod
    def rotate_points(points, rot):
        coords, feats = points[..., :3], points[..., 3:]
        rot = Rotation.from_euler('z', rot).as_matrix()
        if isinstance(points, torch.Tensor):
            rot = points.new_tensor(rot)
            coords = coords @ rot.transpose(-2, -1)
            points = torch.concat((coords, feats), dim=-1)
        else:
            coords = coords @ rot.T
            points = np.concatenate((coords, feats), axis=-1)
        return points

    def random(self, data_dict):
        rot_noise = np.random.uniform(self.range[0], self.range[1])
        return dict(rot=rot_noise)

    def backward(self, data_dict, args):
        args.rot *= -1
        self.forward(data_dict, args)

    def forward(self, data_dict, args):
        if 'points' in data_dict:
            data_dict['points'] = self.rotate_points(data_dict['points'], args.rot)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.rotate_boxes(data_dict['gt_boxes'], args.rot)


@AUGMENTS.register_module('global_translate')
class GlobalTranslate(Augment):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.std = kwargs.get('std', [0, 0, 0])
        assert len(self.std) == 3 and not sum([s < 0 for s in self.std])
        for std in self.std:
            assert std >= 0

    @staticmethod
    def translate_boxes(boxes, trans):
        boxes[:, :3] += trans
        return boxes

    @staticmethod
    def translate_points(points, trans):
        points[:, :3] += trans
        return points

    def random(self, _):
        trans_noise = np.random.normal(scale=self.std, size=3)
        return dict(trans=trans_noise)

    def backward(self, data_dict, args):
        args.trans *= -1
        self.forward(data_dict, -args.trans)

    def forward(self, data_dict, args):
        if 'points' in data_dict:
            data_dict['points'] = self.translate_points(data_dict['points'], args.trans)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.translate_boxes(data_dict['gt_boxes'], args.trans)


@AUGMENTS.register_module('global_scale')
class GlobalScale(Augment):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [0.95, 1.05])
        assert len(self.range) == 2

    @staticmethod
    def scale_boxes(boxes, scale):
        if isinstance(boxes, torch.Tensor):
            scale = boxes.new_tensor(scale)
        boxes[..., :6] *= scale[..., None, None]
        boxes[..., 7:] *= scale[..., None, None]
        return boxes

    @staticmethod
    def scale_points(points, scale):
        if isinstance(points, torch.Tensor):
            scale = points.new_tensor(scale)
        points[..., :3] *= scale[..., None, None]
        return points

    def random(self, _):
        scale_noise = np.random.uniform(self.range[0], self.range[1])
        return dict(scale=scale_noise)

    def backward(self, data_dict, args):
        args.scale = 1.0 / args.scale
        self.forward(data_dict, args)

    def forward(self, data_dict, args):
        if 'points' in data_dict:
            data_dict['points'] = self.scale_points(data_dict['points'], args.scale)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.scale_boxes(data_dict['gt_boxes'], args.scale)


@AUGMENTS.register_module('global_flip')
class GlobalFlip(Augment):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.rotate = GlobalRotate(kwargs)

    def flip_points(self, points):
        points[..., 1] = -points[..., 1]
        return points

    def flip_boxes(self, boxes):
        boxes[..., 1] = -boxes[..., 1]
        boxes[..., 6] = -boxes[..., 6]
        if boxes.shape[-1] > 7: boxes[..., 8] = -boxes[..., 8]
        return boxes

    def random(self, data_dict):
        return self.rotate.random(data_dict)

    def backward(self, data_dict, args):
        if 'points' in data_dict:
            data_dict['points'] = self.flip_points(data_dict['points'])
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.flip_boxes(data_dict['gt_boxes'])
        self.rotate.backward(data_dict, args)

    def forward(self, data_dict, args):
        self.rotate.forward(data_dict, args)
        if 'points' in data_dict:
            data_dict['points'] = self.flip_points(data_dict['points'])
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.flip_boxes(data_dict['gt_boxes'])


#
# @AUGMENTS.register_module('global_sparsify')
# class GlobalSparsify(Transform):
#     def __init__(self, kwargs):
#         super().__init__(kwargs)
#         self.ratio = kwargs.get('ratio', 0.0)
#         assert 1 >= self.ratio >= 0
#
#     def random(self, data_dict):
#         is_keep = np.random.choice([False, True], p=[self.ratio, 1 - self.ratio], size=(data_dict['points'].shape[0]))
#         return dict(is_keep=is_keep)
#
#     def forward(self, data_dict, is_keep):
#         data_dict['points'] = data_dict['points'][is_keep]
#
#
# @AUGMENTS.register_module('background_swap')
# class BackgroundSwap(Transform):
#     def __init__(self, kwargs):
#         super().__init__(kwargs)
#         # we can rotate the background by 180 dg to approximate the background swap
#         # due to only the front 90 dg of scenes is labeled in kitti-like dataset.
#         self.fast_mode = kwargs.get('kitti', True)
#         self.dataset = kwargs.get('dataset', None)
#
#         assert self.fast_mode or self.dataset is None
#
#     def forward(self, data_dict):
#         from ...utils.box_utils import enlarge_box3d
#         from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
#         points = data_dict['points']
#         boxes = data_dict['gt_boxes']
#         coord = points[..., :3]
#         flags = (points_in_boxes_cpu(coord, enlarge_box3d(boxes, [0.2, 0.2, 0.2])) > 0).sum(-1)
#
#         if self.fast_mode:
#             pass
#
#         else:
#             pass
#
#
# @AUGMENTS.register_module('frustum_sparsify')
# class FrustumSparsify(Transform):
#     def __init__(self, kwargs):
#         super().__init__(kwargs)
#         self.range = kwargs.get('range', [-np.pi / 4, np.pi / 4])
#         self.range_v = kwargs.get('range_v', np.pi / 4)
#         self.range_h = kwargs.get('range_h', np.pi / 4)
#         self.distance = kwargs.get('range_r', 35.)
#         self.ratio = kwargs.get('ratio', 0.5)
#
#         assert len(self.range) == 2 and 1 >= self.ratio >= 0
#
#     @staticmethod
#     def get_frustum(yaw, distance, vertical, horizontal):
#         y, z = distance * np.tan([horizontal / 2, vertical / 2])
#         hull = np.array([[0., 0., 0.],
#                          [distance, -y, -z],
#                          [distance, y, -z],
#                          [distance, y, z],
#                          [distance, -y, z]])
#         hull = hull @ Rotation.from_euler('z', yaw).as_matrix().T
#         return hull
#
#     @staticmethod
#     def in_frustum(points, hull):
#         from ...utils.box_utils import in_hull
#         flags = in_hull(points[..., :3], hull)
#         return flags
#
#     def random(self, data_dict):
#         rot_noise = np.random.uniform(self.range[0], self.range[1])
#         frustum = self.get_frustum(rot_noise, self.distance, self.range_v, self.range_h)
#         in_flags = self.in_frustum(data_dict['points'], frustum)
#         num_pts_in_frustum = in_flags.sum()
#         keep_in_frustum = np.random.choice([False, True], p=[self.ratio, 1 - self.ratio], size=num_pts_in_frustum)
#         in_flags[np.where(in_flags)[0][keep_in_frustum]] = 0
#         keep = in_flags == 0
#         return dict(is_keep=keep)
#
#     def forward(self, data_dict, is_keep):
#         data_dict['points'] = data_dict['points'][is_keep]
#
#
# @AUGMENTS.register_module('frustum_noise', 'frustum_jitter')
# class FrustumJitter(Transform):
#     def __init__(self, kwargs):
#         super().__init__(kwargs)
#         self.range = kwargs.get('range', [-np.pi / 4, np.pi / 4])
#         self.range_v = kwargs.get('range_v', np.pi / 4)
#         self.range_h = kwargs.get('range_h', np.pi / 4)
#         self.distance = kwargs.get('range_r', 35.)
#         self.std = kwargs.get('std', 0.7)
#
#         assert len(self.range) == 2 and 1 >= self.std >= 0
#
#     def random(self, data_dict):
#         direction = np.random.uniform(self.range[0], self.range[1])
#         frustum = FrustumSparsify.get_frustum(direction, self.distance, self.range_v, self.range_h)
#         flags = FrustumSparsify.in_frustum(data_dict['points'], frustum)
#         num_pts_in_frustum = flags.sum()
#         noise = np.random.normal(scale=self.std, size=(num_pts_in_frustum, 3))
#         return dict(in_frustum=flags, noise=noise)
#
#     def forward(self, data_dict, in_frustum, noise):
#         data_dict['points'][in_frustum, :3] += noise
#
#
# @AUGMENTS.register_module('box_drop')
# class BoxDrop(Transform):
#     def __init__(self, kwargs):
#         super().__init__(kwargs)
#         self.drop_num = kwargs.get('num', {})
#
#     @staticmethod
#     def remove_points_in_boxes(points, boxes, enlarge=None):
#         from ...utils.box_utils import remove_points_in_boxes3d, enlarge_box3d
#         enlarge = [0.2, 0.2, 0.2] if enlarge is None else enlarge
#         enlarged_boxes = enlarge_box3d(boxes, enlarge)
#         points = remove_points_in_boxes3d(points, enlarged_boxes)
#         return points
#
#     def random(self, data_dict):
#         gt_names = data_dict['gt_names']
#         drop_indices = []
#         for cls, drop_num in self.drop_num.items():
#             cls_flags = cls == gt_names
#             cls_num = cls_flags.sum()
#             lower, upper = np.floor(drop_num), np.ceil(drop_num)
#             drop_num = int(np.random.choice([lower, upper], p=[upper - drop_num, drop_num - lower]))
#             if cls_num <= drop_num or drop_num == 0: continue
#             drop_ind = np.random.choice(np.where(cls_flags)[0], size=int(drop_num), replace=False)
#             drop_indices += list(drop_ind)
#         return dict(drop_which=drop_indices)
#
#     def forward(self, data_dict, drop_which):
#         if drop_which:
#             points, gt_boxes = data_dict['points'], data_dict['gt_boxes']
#             drop_boxes = gt_boxes[drop_which]
#             data_dict['points'] = self.remove_points_in_boxes(points, drop_boxes)
#
#
# @AUGMENTS.register_module('box_paste')
# class BoxPaste(object):
#     def __init__(self, kwargs):
#         from . import database_sampler
#         self.prob = kwargs.get('prob', 0.5)
#         self.verbose = kwargs.get('verbose', False)
#         self.db_sampler = database_sampler.DataBaseSampler(
#             sampler_cfg=kwargs['sampler_cfg'],
#             root_path=kwargs['root_dir'],
#             class_names=kwargs['class_name']
#         )
#
#     def __call__(self, data_dict):
#         self.db_sampler(data_dict)
#         return data_dict
#
#
# @AUGMENTS.register_module('box_rotate', 'local_rotate')
# class BoxRotate(object):
#     pass
#
#
# @AUGMENTS.register_module('box_scale', 'local_scale')
# class BoxScale(object):
#     pass
#
#
# @AUGMENTS.register_module('box_flip', 'local_flip')
# class BoxFlip(object):
#     pass
#
#
# @AUGMENTS.register_module('box_translate', 'local_translate')
# class BoxTranslate(object):
#     pass
#
#
# @AUGMENTS.register_module('box_swap', 'local_swap')
# class BoxSwap(object):
#     pass
#
#
# @AUGMENTS.register_module('part_drop')
# class PartitionDrop(object):
#     pass
#
#
# @AUGMENTS.register_module('part_swap')
# class PartitionSwap(object):
#     pass
#
#
# @AUGMENTS.register_module('part_noise')
# class PartitionNoise(object):
#     pass
#
#
# @AUGMENTS.register_module('part_mix')
# class PartitionMix(object):
#     pass
#
#
# @AUGMENTS.register_module('part_sparsify')
# class PartitionSparsify(object):
#     pass


########################################################################################################################
class Augments:
    def __init__(self, aug_list):
        self.augments: List[Augment] = [AUGMENTS[tf['type']](tf) for tf in aug_list]

    def __call__(self, data_dict, aug_logs=None):
        for aug in self.augments:
            if aug_logs is not None:
                aug(data_dict, aug_logs[aug.name])
            else:
                aug(data_dict)
        return data_dict

    def invert(self, data_dict, aug_logs=None):
        for aug in self.augments[::-1]:
            if aug_logs is not None:
                if aug.name in aug_logs:
                    aug.invert(data_dict, aug_logs[aug.name])
            else:
                aug.invert(data_dict)
        return data_dict
