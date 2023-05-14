import numpy as np
from ...api import log
from ...datasets.augmentor.transforms import AUGMENTOR, Augmentor


@AUGMENTOR.register_module('instance_filling')
class InstanceFilling(Augmentor):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.logger = log.create_logger(name="bg")

        self._bank = kwargs['instance_bank']
        self._bg_miner = kwargs['background_miner']

        self.root_path = self._bank.root_path
        self.class_names = self._bank.class_names

        self.remove_extra_width = kwargs['remove_extra_width']
        self.visualize = kwargs.get('visualize', False)

    @property
    def bank(self):
        return self._bank

    @property
    def bg_infos(self):
        return self._bg_miner.redundant_predicted_boxes

    @property
    def bank_info(self):
        return self._bank.bk_infos

    def get_reliable_background(self, points, unreliable_instances):
        from ...utils.box_utils import remove_points_in_boxes3d, enlarge_box3d
        background = points
        if unreliable_instances.shape[0] > 0:
            enlarge_unreliable = enlarge_box3d(unreliable_instances, self.remove_extra_width)
            background = remove_points_in_boxes3d(points, enlarge_unreliable)
        return background

    def nms_by_volume(self, boxes):
        select = np.zeros(boxes.shape[0], dtype=bool)
        handled = select.copy()
        iou = self.bank.boxes_iou_cpu(boxes, boxes)
        volumes = boxes[:, 3:6].prod(axis=-1)
        box_id = np.argsort(volumes)[::-1]

        for i in box_id:
            if handled[i]:
                continue
            select[i] = True
            handled[iou[i] > 0] = True
        return boxes[select]

    def sample_instances_from_bank(self, num, sample_method='latest'):
        sampled_frame_ids = np.random.choice(list(self.bank_info.keys()), num)

        if sample_method == 'random':
            sampled_ins_ids = [np.random.choice(len(self.bank_info[fid])) for fid in sampled_frame_ids]
        elif sample_method == 'invert_density':
            sampled_ins_ids = []
            for fid in sampled_frame_ids:
                gt_mask = np.array([ins['score'] for ins in self.bank_info[fid]]) < 0
                num_gt = gt_mask.sum()
                num_ins = gt_mask.shape[0]
                assert num_ins != 0

                if num_ins == 1:
                    sampled_ins_ids.append(0)
                else:
                    sample_prob = np.ones_like(gt_mask, dtype=float) * num_gt
                    sample_prob[gt_mask] = num_ins - num_gt
                    try:
                        sampled_ins_ids.append(np.random.choice(num_ins, p=sample_prob / sample_prob.sum()))
                    except ValueError:
                        print(num_ins, sample_prob)
                        sampled_ins_ids.append(num_ins - 1)
        elif sample_method == 'latest':
            sampled_ins_ids = [-1 for _ in sampled_frame_ids]
        else:
            raise NotImplementedError

        names, boxes, points, num_pseudo = [], [], [], 0
        for fid, iid in zip(sampled_frame_ids, sampled_ins_ids):
            info = self.bank_info[fid][iid]
            names.append(info['name'])
            boxes.append(info['box3d_lidar'])
            points.append(self.bank.get_points(path=info['path']))
            num_pseudo += info['score'] >= 0

        self.logger.info(f"sample instances ({num_pseudo} pseudo / {len(names)} total) from bank")
        return names, boxes, points

    def params(self, data_dict):
        frame_id = data_dict['frame_id']
        sampled_names = []
        sampled_boxes = []
        sampled_points = []
        if frame_id in self.bg_infos and len(self.bg_infos[frame_id]) != 0:
            unreliable_instances = self.bg_infos[frame_id]
            holes = self.nms_by_volume(unreliable_instances)
            self.logger.info(f"predicted redundant boxes {len(unreliable_instances)} -> holes {len(holes)}")

            num_filling_boxes = holes.shape[0]
            names, boxes, points = self.sample_instances_from_bank(num_filling_boxes)

            for i in range(num_filling_boxes):
                hole_xyz, hold_volume = holes[i, :3], holes[i, 3:6].prod()
                if boxes[i][3:6].prod() > hold_volume * 1.5:
                    continue
                boxes[i][:3] = hole_xyz
                points[i][:, :3] += hole_xyz
                sampled_names.append(names[i])
                sampled_boxes.append(boxes[i])
                sampled_points.append(points[i])

        return dict(sampled_names=sampled_names, sampled_boxes=sampled_boxes, sampled_points=sampled_points)

    def forward(self, data_dict, sampled_names, sampled_boxes, sampled_points):
        frame_id = data_dict['frame_id']

        if frame_id in self.bg_infos:
            background = self.get_reliable_background(data_dict['points'], self.bg_infos[frame_id])
            data_dict['points'] = np.vstack(sampled_points + [background])
            data_dict['gt_boxes'] = np.vstack(sampled_boxes + [data_dict['gt_boxes']])
            data_dict['gt_names'] = np.vstack(sampled_names + [data_dict['gt_names']]).reshape(-1)
            data_dict['gt_boxes_mask'] = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=bool)

            if self.visualize:
                from ...utils.viz_utils import viz_scenes
                viz_scenes((background, self.nms_by_volume(self.bg_infos[frame_id])),
                           (data_dict['points'], data_dict['gt_boxes']),
                           offset=[0, 35, 0], origin=True)
