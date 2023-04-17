import pickle
import numpy as np
from pathlib import Path

from ...api import on_rank0, create_logger
from ...utils.base import merge_dicts, replace_attr


class Bank:

    def __init__(self, bank_cfg, root_dir=None, class_names=None):
        self.root_path = Path(bank_cfg.get('root_dir', root_dir)).resolve()
        self.class_names = np.array(bank_cfg.get('class_names', class_names))

        self.db_info_path = bank_cfg.db_info_path
        self.bk_info_path = (self.root_path / bank_cfg.bk_info_path).absolute()
        self.pseudo_db_path = (self.root_path / bank_cfg.pseudo_database_path).absolute()
        self.pseudo_db_path.mkdir(exist_ok=True, parents=True)

        self.logger = create_logger()
        self.bk_infos = self.load_from_file()

    @property
    def num_pd_in_db(self):
        return len(list(Path(self.pseudo_db_path).iterdir()))

    @property
    def num_pd_in_bk(self):
        return sum([obj['score'] >= 0 for info in self.bk_infos.values() for obj in info])

    @property
    def num_gt_in_bk(self):
        return sum([obj['score'] < 0 for info in self.bk_infos.values() for obj in info])

    @property
    def num_obj_in_bk(self):
        return sum([len(frame) for frame in self.bk_infos.values()])

    @property
    def valid(self):
        return self.num_pd_in_db == self.num_pd_in_bk

    def load_from_file(self):
        if self.bk_info_path.exists():
            self.bk_infos = pickle.load(open(self.bk_info_path, 'rb'))
        else:
            path = str(self.root_path / self.db_info_path)
            self.bk_infos = self.init_instance_bank(pickle.load(open(path, 'rb')))
        assert self.valid

        self.update_num_gt = self.num_gt_in_bk
        self.update_num_pd = self.num_pd_in_bk
        self.update_num_obj = self.num_obj_in_bk
        self.logger.info(f"Bank loads "
                         f"{len(self.bk_infos)} scenes "
                         f"and {self.update_num_pd}sd+{self.update_num_gt}gt={self.update_num_obj}obj "
                         f"from {self.bk_info_path}")

        if not self.bk_info_path.exists():
            self.save_to_file()
        return self.bk_infos

    @on_rank0
    def save_to_file(self):
        pickle.dump(self.bk_infos, open(self.bk_info_path, 'wb'))
        self.logger.info(f'Bank updates '
                         f'{self.num_pd_in_bk}sd+{self.num_gt_in_bk}gt={self.num_obj_in_bk}obj '
                         f'to {self.bk_info_path}')
        self.update_num_gt = self.num_gt_in_bk
        self.update_num_pd = self.num_pd_in_bk
        self.update_num_obj = self.num_obj_in_bk
        assert self.valid

    @staticmethod
    def init_instance_bank(db_infos):
        bk_infos = {}
        for cls, cls_db_infos in db_infos.items():
            for gt_info in cls_db_infos:
                frame_id = gt_info['image_idx']
                if frame_id not in bk_infos:
                    bk_infos[frame_id] = []
                bk_infos[frame_id].append(gt_info)
        return bk_infos

    def insert_one_instance(self, frame_id, points, box, label, score):
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

    @on_rank0
    def try_insert(self, frame_id, points, pred_boxes, pred_labels, pred_scores):
        from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
        from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

        exist_boxes = np.array([ins['box3d_lidar'] for ins in self.bk_infos[frame_id]])
        iou = boxes_bev_iou_cpu(pred_boxes, exist_boxes)
        # TODO:
        #  we directly refuse all predictions that overlap with objs in bank
        #  without consider to update them by comparing their scores.
        select = np.logical_not(iou.any(axis=-1))
        boxes = pred_boxes[select]
        labels = pred_labels[select]
        scores = pred_scores[select]

        point_flags = points_in_boxes_cpu(points[:, 0:3], boxes)
        for point_flag, box, label, score in zip(point_flags, boxes, labels, scores):
            obj_pts = points[point_flag > 0]
            self.insert_one_instance(frame_id, obj_pts, box, label, score)

    def get_scene(self, frame_id):
        frame = merge_dicts(self.bk_infos[frame_id])
        name = np.array(frame['name'])[..., None]
        labels = np.where(name == self.class_names[None, ...])[1]
        boxes = np.array(frame['box3d_lidar'])
        boxes = np.hstack((boxes, labels[..., None]))
        points = []
        for b, p in zip(boxes, frame['path']):
            points.append(np.fromfile(self.root_path / p, dtype=np.float32).reshape([-1, 4]) + np.array([*b[:3], 0]))
        points = np.vstack(points)
        gt_mask = np.array(frame['score']) < 0
        return points, boxes, gt_mask

    def pseudo_instance_coverage_ratio(self, full_annotated_dataset, threshold_list=(0.1, 0.5, 0.7)):
        from tqdm import tqdm
        from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
        pseudo = self.num_pd_in_bk
        sparse_gt = self.num_gt_in_bk
        full_annotated_gt = 0
        recall_dicts = {k: 0 for k in threshold_list}
        # full_annotated_dataset.kitti_infos = full_annotated_dataset.kitti_infos[:2]
        with replace_attr(full_annotated_dataset.data_augmentor, 'data_augmentor_queue', []):
            with tqdm(iterable=full_annotated_dataset, desc='coverage ratio') as bar:
                for data_dict in full_annotated_dataset:
                    bank_dict = merge_dicts(self.bk_infos[data_dict['frame_id']])
                    pseudo_boxes = np.array(bank_dict['box3d_lidar'])[np.array(bank_dict['score']) > 0]
                    annotated_boxes = data_dict['gt_boxes'][:, :7]

                    full_annotated_gt += annotated_boxes.shape[0]

                    iou = boxes_bev_iou_cpu(pseudo_boxes, annotated_boxes)
                    for thr in threshold_list:
                        recall_dicts[thr] += (iou > thr).any(axis=-1).sum()

                    disp = {str(k): str(int(v)) for k, v in recall_dicts.items()}
                    disp.update(all=str(int(pseudo)))
                    bar.set_postfix(disp)
                    bar.update()
        missing_annotated_num = full_annotated_gt - sparse_gt
        print(f"sparsely annotated instance: {sparse_gt}")
        print(f"fully annotated instance: {full_annotated_gt}")
        print(f"missing annotated instance: {missing_annotated_num}")
        print(f"pseudo instance: {pseudo}")
        print(f"sparse ratio: {sparse_gt / full_annotated_gt}")

        print('\n\t'.join(['recall:'] + [f'{k}-iou: {v / missing_annotated_num}' for k, v in recall_dicts.items()]))
        print('\n\t'.join(['accuracy:'] + [f'{k}-iou: {v / pseudo}' for k, v in recall_dicts.items()]))


def viz(bank, dataset):
    from rd3d.utils import viz_utils
    import open3d
    from pathlib import Path
    with replace_attr(dataset.data_augmentor, 'data_augmentor_queue', []):
        scene_mask = np.array([len(f[1]) for f in sorted(bank.bk_infos.items(), key=lambda x: x[0])]) > 1
        for ind in np.where(scene_mask)[0]:
            data_dict = dataset[ind]
            img_shape = dataset.kitti_infos[ind]['image']['image_shape']
            frame_id = data_dict['frame_id']
            obj_pts, boxes, mask = bank.get_scene(frame_id)
            calib = dataset.get_calib(frame_id)
            pts = dataset.get_lidar(frame_id)
            img = dataset.get_image(frame_id)
            flag = dataset.get_fov_flag(calib.lidar_to_rect(pts[:, 0:3]), img_shape, calib)
            pts = pts[flag]
            pts_img, _ = calib.rect_to_img(calib.lidar_to_rect(pts[:, 0:3]))
            y = np.clip(np.rint(pts_img[:, 1]).astype(int), 0, img.shape[0] - 1)
            x = np.clip(np.round(pts_img[:, 0]).astype(int), 0, img.shape[1] - 1)
            color = img[y, x, :]

            boxes[:, 3:6] += 0.1

            scene = viz_utils.add_scene(title=data_dict['frame_id'], origin=False)
            scene.get_render_option().point_size = 6.0
            scene.get_render_option().background_color = np.ones(3) * 0.8

            viz_utils.add_points(scene, pts, c=color)
            viz_utils.add_boxes(scene, data_dict['gt_boxes'])
            viz_utils.add_boxes(scene, boxes[:, :7])
            viz_utils.add_keypoint(scene, obj_pts, 0.1)

            # viewpoint = Path(__file__).parent / 'open3d_viewpoint.json'
            # params = open3d.io.read_pinhole_camera_parameters(viewpoint.__str__())
            # vc = scene.get_view_control()
            # vc.convert_from_pinhole_camera_parameters(params)

            scene.run()
            del scene

def main():
    from easydict import EasyDict
    from rd3d.datasets import build_dataloader
    from rd3d.api import create_logger
    from configs.base.datasets import kitti_3cls

    kitti_root = 'data/sparse_kitti/'
    bank_cfg = EasyDict(db_info_path='kitti_dbinfos_train.pkl',
                        bk_info_path='ss3d/bkinfos_train.pkl',
                        pseudo_database_path='ss3d/pseudo_database')

    logger = create_logger()
    bank = Bank(bank_cfg, root_dir=kitti_root, class_names=['Car', 'Pedestrian', 'Cyclist'])
    dataset = build_dataloader(kitti_3cls.DATASET, training=True, logger=logger)

    """
    sparsely annotated instance: 3712
    fully annotated instance: 17128
    missing annotated instance: 13416
    pseudo instance: 11301
    sparse ratio: 0.2167211583372256
    recall:
            0.1-iou: 0.6440816935002982
            0.5-iou: 0.6338700059630292
            0.7-iou: 0.5772957662492546
    accuracy:
            0.1-iou: 0.7646225997699319
            0.5-iou: 0.7524997787806389
            0.7-iou: 0.6853375807450668
    """
    bank.pseudo_instance_coverage_ratio(dataset)
    # viz(bank, dataset)


if __name__ == '__main__':
    main()
