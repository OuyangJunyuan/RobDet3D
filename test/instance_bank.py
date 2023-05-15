import unittest
import numpy as np
import pickle
from pathlib import Path
from easydict import EasyDict
from rd3d.runner.ss3d.instance_bank import InstanceBank

analysis_bank = True
visualization = False


class TestInstanceBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = EasyDict(db_info_path='kitti_dbinfos_train.pkl',
                       bk_info_path='test_ss3d/bkinfos_train.pkl',
                       pseudo_database_path='test_ss3d/pseudo_database',
                       root_dir='data/kitti_sparse',
                       class_names=['Car', 'Pedestrian', 'Cyclist'])
        cls.bank = InstanceBank(cfg)

    @unittest.skipIf(analysis_bank, "")
    def test0_insert(self):
        frame_id = "000000"
        points = np.random.rand(10000, 4) * 20
        pred_boxes = np.array([[10, 0, 0, 1, 1, 1, 0], [5, 5, 0, 1, 1, 1, 0.75]])
        pred_labels = np.array([1, 2])
        pred_scores = np.array([0.5, 0.8])
        self.bank.try_insert(frame_id, points, pred_boxes, pred_labels, pred_scores)

    @unittest.skipIf(analysis_bank, "")
    def test1_save_to_disk(self):
        self.bank.save_to_disk()

    @unittest.skipIf(analysis_bank, "")
    def test2_get_scene(self):
        from rd3d.utils import viz_utils

        lidar_path = "/home/nrsl/workspace/temp/RobDet3D/data/kitti_sparse/training/velodyne/000000.bin"
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        obj_pts, ins_boxes, gt_mask = self.bank.get_scene("000000")

        colors = np.array([[1, 0, 0]] * ins_boxes.shape[0])
        colors[gt_mask] = np.array([0, 1, 0])
        viz_utils.viz_scenes((points, (ins_boxes, colors)))


if __name__ == '__main__':
    unittest.main()
