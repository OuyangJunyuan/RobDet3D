import pickle
import numpy as np
from pathlib import Path
from ...datasets.augmentor.transforms import AUGMENTOR, Augmentor
from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu


@AUGMENTOR.register_module('points_filling')
class PointCloudFilling(Augmentor):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.bank = kwargs['bank']
        self.root_dir = Path(kwargs['root_dir'])
        self.pred_infos_path = self.root_dir / kwargs['pred_infos_path']
        self.remove_extra_width = kwargs['remove_extra_width']
        self.class_names = kwargs['class_names']
        self.visualize = kwargs.get('visualize', False)
        if self.pred_infos_path.exists():
            self.pred_infos = pickle.load(open(self.pred_infos_path, 'rb'))
            # print(f"PointCloudFilling: load pred_infos "
            #       f"from {self.pred_infos_path}({self.pred_infos_path.stat().st_size / 10 ** 6}MB)")
        else:
            self.pred_infos = {}
            # print(f"PointCloudFilling: load empty pred_infos")

    def dig_holes_in_scenes(self, points, boxes):
        from ...utils.box_utils import remove_points_in_boxes3d, enlarge_box3d
        return remove_points_in_boxes3d(points, enlarge_box3d(boxes, self.remove_extra_width))

    def fill_holes_in_scenes(self, points, exist_boxes, exist_names, dug_boxes, samples):

        # move sampled boxes to fill the holes.
        fill_boxes = np.array([s['box3d_lidar'] for s in samples])
        fill_boxes[:, :3] = dug_boxes[:, :3]

        # the sampled boxes overlap with gt have been removed in reliable background mining module.
        ego_iou = boxes_bev_iou_cpu(fill_boxes[:, :7], fill_boxes[:, 0:7])
        flag = np.zeros(ego_iou.shape[0], dtype=bool)

        boxes_list, points_list, names_list = [], [], []
        for i in range(flag.size):
            if flag[i]:  # this fill box have been processed
                continue

            names_list.append(samples[i]['name'])

            box = fill_boxes[i]
            boxes_list.append(box)

            path = str(self.root_dir / samples[i]['path'])
            pts = np.fromfile(path, dtype=np.float32).reshape([-1, points.shape[-1]])
            pts[:, :3] += box[:3]
            points_list.append(pts)

            flag[ego_iou[i] > 0] = True  # only one fill box is selected for each overlapped hole

        boxes = np.vstack(boxes_list)

        # TODO: remove the points in boxes or remove the boxes directly to avoid physical meaningless.
        points = self.dig_holes_in_scenes(points, boxes)  # empty the points in boxes to place new points
        points = np.vstack([points] + points_list)

        boxes = np.vstack([exist_boxes, boxes])

        names = np.hstack([exist_names, np.array(names_list)])
        return points, boxes, names

    def params(self, data_dict):
        boxes = self.pred_infos.get(data_dict['frame_id'], np.zeros([0, 7]))
        random_sample_frame_id = np.random.choice(list(self.bank.bk_infos.keys()), boxes.shape[0])
        random_sample_obj_id = [np.random.choice(len(self.bank.bk_infos[fid])) for fid in random_sample_frame_id]
        return dict(frame_ids=random_sample_frame_id, obj_ind=random_sample_obj_id)

    def forward(self, data_dict, args):
        frame_id = data_dict['frame_id']
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']
        if frame_id in self.pred_infos:
            unreliable_objs = self.pred_infos[frame_id]
            if unreliable_objs.shape[0] > 0:
                points = self.dig_holes_in_scenes(points, unreliable_objs)
                sampled_bank_objs = [self.bank.bk_infos[fid][ind] for fid, ind in zip(args.frame_ids, args.obj_ind)]
                points, gt_boxes, gt_names = self.fill_holes_in_scenes(points, gt_boxes, gt_names,
                                                                       unreliable_objs, sampled_bank_objs)

                gt_boxes_mask = [n in self.class_names for n in gt_names]
                if self.visualize:
                    from ...utils.viz_utils import viz_scenes
                    viz_scenes((data_dict['points'], data_dict['gt_boxes']),
                               (data_dict['points'], unreliable_objs),
                               (points, gt_boxes),
                               offset=[0, 35, 0], origin=True)
                data_dict.update(gt_boxes=gt_boxes, gt_names=gt_names, points=points, gt_boxes_mask=gt_boxes_mask)
