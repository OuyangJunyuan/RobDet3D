import pickle
import numpy as np
from pathlib import Path

from ...api import on_rank0, create_logger
from ...utils.base import merge_dicts


class InstanceBank(object):

    def __init__(self, bank_cfg, root_dir=None, class_names=None):
        self.bk_infos = {}
        self.previous_num_total = 0
        self.previous_num_pseudo = 0
        self.previous_num_anno = 0

        self.root_path = Path(bank_cfg.get('root_dir', root_dir)).resolve()
        self.class_names = np.array(bank_cfg.get('class_names', class_names))
        self.db_info_path = (self.root_path / bank_cfg.db_info_path).absolute()
        self.bk_info_path = (self.root_path / bank_cfg.bk_info_path).absolute()
        self.pseudo_db_path = (self.root_path / bank_cfg.pseudo_database_path).absolute()
        self.pseudo_db_path.mkdir(exist_ok=True, parents=True)

        self.logger = create_logger("bank", log_file=self.pseudo_db_path / "../log.txt", stderr=False)
        self.logger.warning("*************** InstanceBank ***************")

        self.load_from_disk()
        if not self.bk_info_path.exists():
            self.save_to_disk()

        self.logger.info(msg=dict(class_names=self.class_names,
                                  num_scenes=len(self.bk_infos),
                                  num_annotated=self.num_annotated_labels_in_infos,
                                  num_pseudo=self.num_pseudo_labels_in_infos,
                                  num_total=self.num_labels_in_disk,
                                  db_info_path=self.db_info_path,
                                  bk_info_path=self.bk_info_path,
                                  pseudo_db_path=self.pseudo_db_path),
                         title=['bank', 'value'])

    @property
    def num_pseudo_labels_in_disk(self):
        return len(list(Path(self.pseudo_db_path).iterdir()))

    @property
    def num_labels_in_disk(self):
        return sum([len(frame) for frame in self.bk_infos.values()])

    @property
    def num_annotated_labels_in_infos(self):
        return sum([obj['score'] < 0 for info in self.bk_infos.values() for obj in info])

    @property
    def num_pseudo_labels_in_infos(self):
        return sum([obj['score'] >= 0 for scene in self.bk_infos.values() for obj in scene])

    @staticmethod
    def init_from_gt(db_infos):
        bk_infos = {}
        for cls, cls_db_infos in db_infos.items():
            for gt_info in cls_db_infos:
                frame_id = gt_info['image_idx']
                if frame_id not in bk_infos:
                    bk_infos[frame_id] = []
                bk_infos[frame_id].append(gt_info)
        return bk_infos

    def load_from_disk(self):
        if self.bk_info_path.exists():
            with open(self.bk_info_path, 'rb') as f:
                self.bk_infos = pickle.load(f)
            self.previous_num_total = self.num_labels_in_disk
            self.previous_num_pseudo = self.num_pseudo_labels_in_infos
            self.previous_num_anno = self.num_annotated_labels_in_infos
            self.logger.info(f"load information from {self.bk_info_path}")
        else:
            with open(self.db_info_path, 'rb') as f:
                self.bk_infos = self.init_from_gt(pickle.load(f))
            self.logger.info(f"init information from {self.db_info_path}")

        assert self.num_pseudo_labels_in_disk == self.num_pseudo_labels_in_infos

    @on_rank0
    def save_to_disk(self):
        assert self.num_pseudo_labels_in_disk == self.num_pseudo_labels_in_infos
        with open(self.bk_info_path, 'wb') as f:
            pickle.dump(self.bk_infos, f)

        num_update = self.num_pseudo_labels_in_disk - self.previous_num_pseudo
        self.logger.info(f"update {num_update} pseudo instances and save information to {self.bk_info_path}")

        self.previous_num_anno = self.num_annotated_labels_in_infos
        self.previous_num_pseudo = self.num_pseudo_labels_in_infos
        self.previous_num_total = self.num_labels_in_disk

    @staticmethod
    def boxes_iou_cpu(boxes1, boxes2):
        from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
        return boxes_bev_iou_cpu(boxes1, boxes2)

    def check_collision_free(self, frame_id, query_boxes):
        if frame_id not in self.bk_infos:
            return np.ones(query_boxes.shape[0], dtype=bool)
        exist_boxes = np.array([ins['box3d_lidar'] for ins in self.bk_infos[frame_id]])
        iou = self.boxes_iou_cpu(query_boxes, exist_boxes)
        collision_free = np.logical_not(iou.any(axis=-1))
        return collision_free

    @on_rank0
    def insert_one_instance(self, frame_id, points, box, label, score):
        if frame_id not in self.bk_infos:
            self.logger.warning(f"creat new scenes {frame_id}")
            self.bk_infos[frame_id] = []

        points[:, :3] -= box[None, :3]

        name = self.class_names[label - 1]
        idx = self.bk_infos[frame_id].__len__()
        filename = '%s_%s_%d.bin' % (frame_id, name, idx)
        filepath = self.pseudo_db_path / filename

        with open(filepath, 'w') as f:
            points.tofile(f)

        db_path = str(filepath.relative_to(self.root_path))
        db_info = {'name': name, 'path': db_path, 'image_idx': frame_id, 'gt_idx': idx,
                   'box3d_lidar': box, 'num_points_in_gt': points.shape[0],
                   'difficulty': 0, 'bbox': [0, 0, 1, 1], 'score': score}
        self.bk_infos[frame_id].append(db_info)

        if len(points) < 5:
            self.logger.warning(f"instance {idx} with only {len(points)} points")
        self.logger.warning(f"add instance {idx} to scene {frame_id} (class: {name})")

    @on_rank0
    def try_insert(self, frame_id, points, pred_boxes, pred_labels, pred_scores, **kwargs):
        from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
        # TODO: we directly refuse all predictions that overlap with objs in bank
        #  without consider to update them by comparing their scores.
        if len(pred_boxes) == 0:
            return
        select = self.check_collision_free(frame_id, pred_boxes)

        boxes = pred_boxes[select]
        labels = pred_labels[select]
        scores = pred_scores[select]

        point_flags = points_in_boxes_cpu(points[:, 0:3], boxes)
        for point_flag, box, label, score in zip(point_flags, boxes, labels, scores):
            obj_pts = points[point_flag > 0]
            # if obj_pts.shape[0] > 0:
            self.insert_one_instance(frame_id, obj_pts, box, label, score)

    def get_points(self, frame_id=None, ins_id=None, path=None):
        points_path = self.root_path / (path or self.bk_infos[frame_id][ins_id]['path'])
        return np.fromfile(points_path, dtype=np.float32).reshape([-1, 4])

    def get_scene(self, frame_id, return_points=True):
        frame = merge_dicts(self.bk_infos[frame_id])
        name = np.array(frame['name'])[..., None]
        labels = np.where(name == self.class_names[None, ...])[1] + 1
        boxes = np.array(frame['box3d_lidar'])
        boxes = np.hstack((boxes, labels[..., None]))
        gt_mask = np.array(frame['score']) < 0
        if return_points:
            points = [self.get_points(path=p) + np.array([*b[:3], 0]) for b, p in zip(boxes, frame['path'])]
            return points, boxes, gt_mask
        else:
            return boxes, gt_mask
