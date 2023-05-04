import torch
import numpy as np

from scipy.spatial.transform import Rotation
from ...utils.base import Register, merge_dicts

AUGMENTOR = Register('AUGMENTS')


def enable(prob, size=None):
    i_size = int(size) if size is not None else None
    mask = np.random.choice([False, True], p=[1 - prob, prob], size=i_size)
    if isinstance(size, torch.Tensor):
        mask = size.new_tensor(mask, dtype=torch.bool)
    return mask


def angle_out_of_range(x, mid, width, handle=np):
    diff = (x - mid + np.pi) % (2 * np.pi) - np.pi
    width /= 2
    return handle.logical_or(diff < -width, width < diff)


def try_points_in_boxes_masks_from_cache(data_dict):
    if 'points_in_boxes' in data_dict['augment_logs']:
        mask = data_dict['augment_logs']['points_in_boxes']
    else:
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']
        if isinstance(points, torch.Tensor):
            from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
            flag = points_in_boxes_gpu(points[None, ..., 0:3], gt_boxes[None, ...])[0].long()
            mask = flag.new_zeros([gt_boxes.shape[0] + 1, points.shape[0]], dtype=torch.bool)
            index = torch.nonzero(flag + 2)[..., 0]
            mask[flag, index] = True
            mask = mask[:gt_boxes.shape[0]].contiguous()
        else:
            from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
            mask = points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()
            mask = mask == 1
        data_dict['augment_logs']['points_in_boxes'] = mask
    return mask


class Augmentor:
    def __init__(self, kwargs):
        self.name = type(self).__name__
        self.prob = kwargs.get('prob', 1.0)

    def build_params(self, params):
        from easydict import EasyDict
        if isinstance(params, list):
            params = merge_dicts(params)
        # params = {k: np.array(v) for k, v in params.items()}
        return EasyDict(params)

    def invert(self, data_dict, params=None):
        params = self.build_params(params or data_dict['augment_logs'][self.name])
        assert hasattr(self, 'backward')
        self.backward(data_dict, **params)
        return data_dict

    def __call__(self, data_dict, params=None):
        if 'augment_logs' not in data_dict:
            data_dict['augment_logs'] = {}
        if enable(self.prob):
            params = self.build_params(params or self.params(data_dict))
            self.forward(data_dict, **params)
            data_dict['augment_logs'][self.name] = params
        return data_dict


@AUGMENTOR.register_module('global_rotate')
class GlobalRotate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [-np.pi / 4, np.pi / 4])

        if not isinstance(self.range, list):
            self.range = [abs(self.range)]
        if len(self.range) == 1:
            self.range = [-self.range[0], self.range[0]]

    @staticmethod
    def rotate_points(points, rot):
        def rotate_points_numpy(points, rot):
            coords, feats = points[..., :3], points[..., 3:]
            rot = Rotation.from_euler('z', rot).as_matrix()
            coords = coords @ rot.T
            points = np.concatenate((coords, feats), axis=-1)
            return points

        def rotate_points_torch(points, rot):
            coords, feats = points[..., :3], points[..., 3:]
            rot = Rotation.from_euler('z', rot).as_matrix()
            rot = points.new_tensor(rot)
            coords = coords @ rot.transpose(-2, -1)
            points = torch.concat((coords, feats), dim=-1)
            return points

        if isinstance(points, torch.Tensor):
            return rotate_points_torch(points, rot)
        else:
            return rotate_points_numpy(points, rot)

    @staticmethod
    def rotate_boxes(boxes, rot, globally=True):
        def rotate_boxes_numpy(boxes, rot):
            if globally:
                boxes[..., :3] = GlobalRotate.rotate_points(boxes[..., :3], rot)
            boxes[..., 6] = boxes[..., 6] + rot
            if boxes.shape[-1] > 7:
                velo_xy = np.concatenate([boxes[..., 7:9], np.zeros_like(boxes[..., :1])], axis=-1)
                boxes[..., 7:9] = GlobalRotate.rotate_points(velo_xy, rot)[..., :2]
            return boxes

        def rotate_boxes_torch(boxes, rot):
            if globally:
                boxes[..., :3] = GlobalRotate.rotate_points(boxes[..., :3], rot)
            rot = boxes.new_tensor(rot)
            boxes[..., 6] = boxes[..., 6] + rot
            if boxes.shape[-1] > 7:
                velo_xy = torch.concat([boxes[..., 7:9], torch.zeros_like(boxes[..., :1])], dim=-1)
                boxes[..., 7:9] = GlobalRotate.rotate_points(velo_xy, rot)[..., :2]
            return boxes

        if isinstance(boxes, torch.Tensor):
            return rotate_boxes_torch(boxes, rot)
        else:
            return rotate_boxes_numpy(boxes, rot)

    def params(self, data_dict):
        rot_noise = np.array(np.random.uniform(self.range[0], self.range[1]))
        return dict(rot=rot_noise)

    def backward(self, data_dict, rot):
        self.forward(data_dict, -rot)

    def forward(self, data_dict, rot):
        if 'points' in data_dict:
            data_dict['points'] = self.rotate_points(data_dict['points'], rot)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.rotate_boxes(data_dict['gt_boxes'], rot)


@AUGMENTOR.register_module('global_translate')
class GlobalTranslate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.std = kwargs.get('std', [0, 0, 0])
        if not isinstance(self.std, list):
            self.std = [abs(self.std)]
        if len(self.std) == 1:
            self.std *= 3

    @staticmethod
    def translate_boxes(boxes, trans):
        if isinstance(boxes, torch.Tensor):
            trans = boxes.new_tensor(trans)
        boxes[:, :3] += trans
        return boxes

    @staticmethod
    def translate_points(points, trans):
        if isinstance(points, torch.Tensor):
            trans = points.new_tensor(trans)
        points[:, :3] += trans
        return points

    def params(self, data_dict):
        trans_noise = np.random.normal(scale=self.std, size=3)
        return dict(trans=trans_noise)

    def backward(self, data_dict, trans):
        self.forward(data_dict, -trans)

    def forward(self, data_dict, trans):
        if 'points' in data_dict:
            data_dict['points'] = self.translate_points(data_dict['points'], trans)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.translate_boxes(data_dict['gt_boxes'], trans)


@AUGMENTOR.register_module('global_scale')
class GlobalScale(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [0.95, 1.05])
        if not isinstance(self.range, list):
            self.range = [abs(self.range)]
        if len(self.range) == 1:
            self.range = [1 - self.range[0], 1 + self.range[0]]

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

    def params(self, data_dict):
        scale_noise = np.array(np.random.uniform(self.range[0], self.range[1]))
        return dict(scale=scale_noise)

    def backward(self, data_dict, scale):
        self.forward(data_dict, 1.0 / scale)

    def forward(self, data_dict, scale):
        if 'points' in data_dict:
            data_dict['points'] = self.scale_points(data_dict['points'], scale)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.scale_boxes(data_dict['gt_boxes'], scale)


@AUGMENTOR.register_module('global_flip')
class GlobalFlip(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.axis = kwargs.get('axis', ['x'])
        if not isinstance(self.axis, list):
            self.axis = [self.axis]

    @staticmethod
    def flip_points(points, axis_x, axis_y):
        if axis_x:
            points[..., 1] = -points[..., 1]
        if axis_y:
            points[..., 0] = -points[..., 0]
        return points

    @staticmethod
    def flip_boxes(boxes, axis_x, axis_y, globally=True):
        if axis_x:
            if globally:
                boxes[..., 1] = -boxes[..., 1]
            boxes[..., 6] = -boxes[..., 6]
            if boxes.shape[-1] > 7:
                boxes[..., 8] = boxes[..., 8]
        if axis_y:
            if globally:
                boxes[..., 0] = -boxes[..., 0]
            boxes[..., 6] = -boxes[..., 6] + np.pi
            if boxes.shape[-1] > 7:
                boxes[..., 7] = boxes[..., 7]
        return boxes

    def params(self, data_dict):
        return {k: enable(0.5) if k in self.axis else False for k in ['x', 'y']}

    def backward(self, data_dict, x, y):
        self.forward(data_dict, x, y)

    def forward(self, data_dict, x, y):
        if 'points' in data_dict:
            data_dict['points'] = self.flip_points(data_dict['points'], x, y)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.flip_boxes(data_dict['gt_boxes'], x, y)


@AUGMENTOR.register_module('global_sparsify')
class GlobalSparsify(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.ratio = abs(kwargs.get('keep_ratio', 0.0))

    def params(self, data_dict):
        points = data_dict['points']
        if len(points.shape) == 2:
            num_points = points.shape[0]
        else:
            raise ValueError
        mask = enable(self.ratio, size=num_points)
        mask_inv = np.logical_not(mask)
        drop_part = points[mask_inv]
        return dict(mask=mask, mask_inv=mask_inv, drop_part=drop_part)

    def backward(self, data_dict, mask, mask_inv, drop_part):
        points = data_dict['points']
        num_points = mask.shape[0]
        if isinstance(points, torch.Tensor):
            raw_points = points.new_zeros([num_points, points.shape[-1]])
            raw_points[mask] = points
            raw_points[mask_inv] = points.new_tensor(drop_part)
        else:
            raw_points = np.zeros([num_points, points.shape[-1]])
            raw_points[mask] = data_dict['points']
            raw_points[mask_inv] = drop_part
        data_dict['points'] = raw_points

    def forward(self, data_dict, mask, mask_inv, drop_part):
        data_dict['points'] = data_dict['points'][mask]


@AUGMENTOR.register_module('frustum_sparsify')
class FrustumSparsify(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.head = kwargs.get('direction', [-np.pi / 4, np.pi / 4])
        self.range = kwargs.get('range', np.pi / 4)
        self.ratio = np.clip(abs(kwargs.get('keep_ratio', 0.5)), 0, 1)
        assert len(self.head) == 2

    def params(self, data_dict):
        points = data_dict['points']
        handle = torch if isinstance(points, torch.Tensor) else np

        head = np.random.uniform(self.head[0], self.head[1])
        width = abs(np.random.normal(scale=np.pi / 4))
        points_head = handle.arctan2(points[..., 1], points[..., 0])
        out_range = angle_out_of_range(points_head, head, width, handle)

        in_range = handle.logical_not(out_range)
        out_range[in_range] = enable(self.ratio, size=in_range.sum())
        mask = out_range
        mask_inv = handle.logical_not(mask)
        drop_part = data_dict['points'][mask_inv]
        return dict(mask=mask, mask_inv=mask_inv, drop_part=drop_part)

    def backward(self, data_dict, mask, mask_inv, drop_part):
        points = data_dict['points']
        num_points = mask.shape[0]
        if isinstance(points, torch.Tensor):
            raw_points = points.new_zeros([num_points, points.shape[-1]])
            raw_points[mask] = points
            raw_points[mask_inv] = points.new_tensor(drop_part)
        else:
            raw_points = np.zeros([num_points, points.shape[-1]])
            raw_points[mask] = data_dict['points']
            raw_points[mask_inv] = drop_part

        data_dict['points'] = raw_points

    def forward(self, data_dict, mask, mask_inv, drop_part):
        data_dict['points'] = data_dict['points'][mask]


@AUGMENTOR.register_module('frustum_noise', 'frustum_jitter')
class FrustumJitter(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.head = kwargs.get('direction', [-np.pi / 4, np.pi / 4])
        self.range = kwargs.get('range', np.pi / 4)
        self.std = abs(kwargs.get('std', 0.5))

    def params(self, data_dict):
        points = data_dict['points']
        handle = torch if isinstance(points, torch.Tensor) else np

        head = np.random.uniform(self.head[0], self.head[1])
        width = abs(np.random.normal(scale=np.pi / 4))
        points_head = handle.arctan2(points[..., 1], points[..., 0])
        out_range = angle_out_of_range(points_head, head, width, handle)
        in_range = handle.logical_not(out_range)
        if handle is np:
            loc_noise = np.random.normal(scale=self.std, size=(in_range.sum(), 3))
        else:
            loc_noise = torch.randn(size=(in_range.sum(), 3), device=points.device) * self.std
        return dict(mask=in_range, loc_noise=loc_noise)

    def backward(self, data_dict, mask, loc_noise):
        self.forward(data_dict, mask, -loc_noise)

    def forward(self, data_dict, mask, loc_noise):
        points = data_dict['points']
        if isinstance(points, torch.Tensor):
            mask = points.new_tensor(mask, dtype=torch.bool)
            loc_noise = points.new_tensor(loc_noise)
        data_dict['points'][mask, :3] += loc_noise


@AUGMENTOR.register_module('box_rotate', 'local_rotate')
class BoxRotate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [-np.pi / 4, np.pi / 4])

        if not isinstance(self.range, list):
            self.range = [abs(self.range)]
        if len(self.range) == 1:
            self.range = [-self.range[0], self.range[0]]

    @staticmethod
    def rotate_boxes_local(boxes, rot):
        boxes = GlobalRotate.rotate_boxes(boxes, rot, globally=False)
        return boxes

    @staticmethod
    def rotate_points_local(points, rots, masks, boxes):
        for mask, box, rot in zip(masks, boxes, rots):
            offset = box[:3]
            points_of_box = points[mask]
            points_of_box[:, :3] -= offset
            points_of_box = GlobalRotate.rotate_points(points_of_box, rot)
            points_of_box[:, :3] += offset
            points[mask] = points_of_box
        return points

    def params(self, data_dict):
        masks = try_points_in_boxes_masks_from_cache(data_dict)
        rot_noise = np.random.uniform(self.range[0], self.range[1], data_dict['gt_boxes'].shape[0])
        return dict(masks=masks, rot_noise=rot_noise)

    def backward(self, data_dict, masks, rot_noise):
        self.forward(data_dict, masks, -rot_noise)

    def forward(self, data_dict, masks, rot_noise):
        data_dict['points'] = self.rotate_points_local(data_dict['points'], rot_noise, masks, data_dict['gt_boxes'])
        data_dict['gt_boxes'] = self.rotate_boxes_local(data_dict['gt_boxes'], rot_noise)


@AUGMENTOR.register_module('box_translate', 'local_translate')
class BoxTranslate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.std = kwargs.get('std', [0, 0, 0])
        if not isinstance(self.std, list):
            self.std = [abs(self.std)]
        if len(self.std) == 1:
            self.std *= 3

    @staticmethod
    def translate_boxes_local(boxes, trans):
        return GlobalTranslate.translate_boxes(boxes, trans)

    @staticmethod
    def translate_points_local(points, trans, masks, boxes):
        if isinstance(points, torch.Tensor):
            offsets = torch.zeros_like(points[..., :3])
            trans = points.new_tensor(trans)
        else:
            offsets = np.zeros_like(points[..., :3])
        for mask, offset in zip(masks, trans):
            offsets[mask] = offset
        points[..., :3] += offsets
        return points

    def params(self, data_dict):
        masks = try_points_in_boxes_masks_from_cache(data_dict)
        trans_noise = np.random.normal(scale=self.std, size=[data_dict['gt_boxes'].shape[0], 3])
        return dict(masks=masks, trans_noise=trans_noise)

    def backward(self, data_dict, masks, trans_noise):
        self.forward(data_dict, masks, -trans_noise)

    def forward(self, data_dict, masks, trans_noise):
        data_dict['points'] = self.translate_points_local(data_dict['points'], trans_noise, masks,
                                                          data_dict['gt_boxes'])
        data_dict['gt_boxes'] = self.translate_boxes_local(data_dict['gt_boxes'], trans_noise)


@AUGMENTOR.register_module('box_scale', 'local_scale')
class BoxScale(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [0.95, 1.05])
        if not isinstance(self.range, list):
            self.range = [abs(self.range)]
        if len(self.range) == 1:
            self.range = [1 - self.range[0], 1 + self.range[0]]

    @staticmethod
    def scale_boxes_local(boxes, scale):
        if isinstance(boxes, torch.Tensor):
            scale = boxes.new_tensor(scale)
        boxes[..., 3:6] *= scale[..., None]
        return boxes

    @staticmethod
    def scale_points_local(points, scales, masks, boxes):
        for mask, box, scale in zip(masks, boxes, scales):
            offset = box[:3]
            points_of_box = points[mask]
            points_of_box[:, :3] -= offset
            points_of_box = GlobalScale.scale_points(points_of_box, scale)
            points_of_box[:, :3] += offset
            points[mask] = points_of_box
        return points

    def params(self, data_dict):
        scale_noise = np.random.uniform(self.range[0], self.range[1], data_dict['gt_boxes'].shape[0])
        masks = try_points_in_boxes_masks_from_cache(data_dict)
        return dict(masks=masks, scale_noise=scale_noise)

    def backward(self, data_dict, masks, scale_noise):
        self.forward(data_dict, masks, 1.0 / scale_noise)

    def forward(self, data_dict, masks, scale_noise):
        data_dict['points'] = self.scale_points_local(data_dict['points'], scale_noise, masks, data_dict['gt_boxes'])
        data_dict['gt_boxes'] = self.scale_boxes_local(data_dict['gt_boxes'], scale_noise)


@AUGMENTOR.register_module('box_flip', 'local_flip')
class BoxFlip(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.axis = kwargs.get('axis', ['x'])
        if not isinstance(self.axis, list):
            self.axis = [self.axis]

    @staticmethod
    def flip_points_local(points, axes_x, axes_y, masks, boxes):
        for mask, box, a_x, a_y in zip(masks, boxes, axes_x, axes_y):
            offset = box[:3]
            points_of_box = points[mask]
            points_of_box[:, :3] -= offset
            points_of_box = GlobalFlip.flip_points(points_of_box, a_x, a_y)
            points_of_box[:, :3] += offset
            points[mask] = points_of_box
        return points

    @staticmethod
    def flip_boxes_local(boxes, axes_x, axes_y):
        for i in range(boxes.shape[0]):
            boxes[i] = GlobalFlip.flip_boxes(boxes[i], axes_x[i], axes_y[i], globally=False)
        return boxes

    def params(self, data_dict):
        params = {k: enable(0.5, size=data_dict['gt_boxes'].shape[0]) if k in self.axis else False for k in ['x', 'y']}
        params.update(masks=try_points_in_boxes_masks_from_cache(data_dict))
        return params

    def backward(self, data_dict, masks, x, y):
        self.forward(data_dict, masks, x, y)

    def forward(self, data_dict, masks, x, y):
        data_dict['points'] = self.flip_points_local(data_dict['points'], x, y, masks, data_dict['gt_boxes'])
        data_dict['gt_boxes'] = self.flip_boxes_local(data_dict['gt_boxes'], x, y)


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
#
# @AUGMENTS.register_module('box_swap', 'local_swap')
# class BoxSwap(object):
#     pass
# @AUGMENTOR.register_module('box_drop')
# class BoxDrop(Augmentor):
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


########################################################################################################################
class AugmentorList:
    def __init__(self, aug_list):
        self.augmentor_list = [AUGMENTOR.from_cfg(tf) for tf in aug_list]

    def __call__(self, data_dict, aug_logs={}):
        for aug in self.augmentor_list:
            aug(data_dict, aug_logs.get(aug.name, None))
        return data_dict

    def invert(self, data_dict, aug_logs={}):
        for aug in self.augmentor_list[::-1]:
            aug.invert(data_dict, aug_logs.get(aug.name, None))
        return data_dict
