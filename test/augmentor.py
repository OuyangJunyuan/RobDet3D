import unittest
import torch
import numpy as np

from rd3d.datasets.augmentor.transforms import AUGMENTOR, AugmentorList
from rd3d.utils.viz_utils import viz_scenes

forward_type = 'cpu'
backward_type = 'cuda'
exam_each = True
exam_invert = False
exam_all = False
visualization = True


class TestAugmentor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from rd3d import build_dataloader
        from rd3d.api.config import Config
        dataset_config = Config.fromfile("configs/base/datasets/kitti_3cls.py").DATASET
        dataset_config.DATA_AUGMENTOR.AUG_CONFIG_LIST = []
        cls.dataset = dataset = build_dataloader(dataset_config)
        cls.data_dict = data_dict = dataset[-1]

        data_dict["gt_boxes"] = torch.from_numpy(data_dict["gt_boxes"][..., :7])
        data_dict["points"] = torch.from_numpy(data_dict["points"])
        if forward_type == 'cuda':
            data_dict["gt_boxes"] = data_dict["gt_boxes"].cuda()
            data_dict["points"] = data_dict["points"].cuda()
        cls.raw_points = data_dict["points"].clone()
        cls.raw_boxes = data_dict["gt_boxes"].clone()

    def tearDown(self) -> None:
        if visualization:
            viz_scenes((self.raw_points, self.raw_boxes),
                       (self.data_dict["points"], self.data_dict["gt_boxes"]),
                       offset=[0, 30, 0], origin=True, title=self._testMethodName)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test0_global_rotate(self):
        global_rotate = AUGMENTOR.from_cfg(dict(name='global_rotate', prob=1.0, range=[-np.pi / 4, np.pi / 4]))
        global_rotate(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test1_global_translate(self):
        global_translate = AUGMENTOR.from_cfg(dict(name='global_translate', prob=1.0, std=[10, 10, 10]))
        global_translate(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test2_global_scale(self):
        global_scale = AUGMENTOR.from_cfg(dict(name='global_scale', prob=1.0, range=[0.2, 0.5]))
        global_scale(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test3_global_flip(self):
        global_flip = AUGMENTOR.from_cfg(dict(name='global_flip', prob=1.0, axis=['x']))
        global_flip(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test5_global_sparsity(self):
        global_sparsify = AUGMENTOR.from_cfg(dict(name='global_sparsify', prob=1.0, keep_ratio=0.1))
        global_sparsify(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test6_frustum_sparsify(self):
        frustum_sparsify = AUGMENTOR.from_cfg(
            dict(name='frustum_sparsify', prob=1.0, direction=[0, 0], range=np.pi / 4, keep_ratio=0.05)
        )
        frustum_sparsify(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test7_frustum_noise(self):
        frustum_noise = AUGMENTOR.from_cfg(
            dict(name='frustum_noise', prob=1.0, direction=[0, 0], range=np.pi / 4, std=10.2)
        )
        frustum_noise(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test8_box_rotate(self):
        box_rotate = AUGMENTOR.from_cfg(
            dict(name='box_rotate', prob=1.0, range=[-np.pi / 4, np.pi / 4])
        )
        box_rotate(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test9_box_translate(self):
        box_translate = AUGMENTOR.from_cfg(
            dict(name='box_translate', prob=1.0, std=[1, 1, 10])
        )
        box_translate(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test90_box_scale(self):
        box_scale = AUGMENTOR.from_cfg(
            dict(name='box_scale', prob=1.0, range=[0.5, 2])
        )
        box_scale(self.data_dict)

    @unittest.skipUnless(exam_each, 'exam_each')
    def test91_box_flip(self):
        box_flip = AUGMENTOR.from_cfg(
            dict(name='box_flip', prob=1.0, axis=['x'])
        )
        box_flip(self.data_dict)

    @unittest.skipUnless(exam_invert, "exam_invert")
    def test999_compose(self):
        aus = AugmentorList([
            dict(name='global_rotate', prob=1.0, range=[-1.78539816, 1.78539816]),
            dict(name='global_translate', prob=1.0, std=[20, 20, 0]),
            dict(name='global_scale', prob=1.0, range=[0.5, 2.0]),
            dict(name='global_flip', prob=1.0, axis=['x', 'y']),
            dict(name='global_sparsify', prob=1.0, keep_ratio=0.5),
            dict(name='frustum_sparsify', prob=1.0, direction=[0, 0], range=np.pi / 4, keep_ratio=0.5),
            dict(name='frustum_noise', prob=1.0, direction=[0, 0], range=np.pi / 4, std=2.2),
            dict(name='box_rotate', prob=1.0, range=[-np.pi / 4, np.pi / 4]),
            dict(name='box_translate', prob=1.0, std=[1, 1, 10]),
            dict(name='box_scale', prob=1.0, range=[0.8, 1.2]),
            dict(name='box_flip', prob=1.0, axis=['x', 'y'])
        ])
        aus(self.data_dict)
        if backward_type != 'cuda':
            self.data_dict["gt_boxes"] = self.data_dict["gt_boxes"].cpu()
            self.data_dict["points"] = self.data_dict["points"].cpu()
        elif forward_type != 'cuda':
            self.data_dict["gt_boxes"] = torch.from_numpy(self.data_dict["gt_boxes"]).cuda()
            self.data_dict["points"] = torch.from_numpy(self.data_dict["points"]).cuda()
        aus.invert(self.data_dict)

    @unittest.skipUnless(exam_all, "skip for no examing")
    def test9999_exam(self):
        from tqdm import tqdm
        aus = AugmentorList([
            dict(name='global_rotate', prob=1.0, range=[-1.78539816, 1.78539816]),
            dict(name='global_translate', prob=1.0, std=[20, 20, 0]),
            dict(name='global_scale', prob=1.0, range=[0.5, 2.0]),
            dict(name='global_flip', prob=1.0, axis=['x', 'y']),
            dict(name='global_sparsify', prob=1.0, keep_ratio=0.5),
            dict(name='frustum_sparsify', prob=1.0, direction=[0, 0], range=np.pi / 4, keep_ratio=0.5),
            dict(name='frustum_noise', prob=1.0, direction=[0, 0], range=np.pi / 4, std=2.2),
            dict(name='box_rotate', prob=1.0, range=[-np.pi / 4, np.pi / 4]),
            dict(name='box_translate', prob=1.0, std=[1, 1, 10]),
            dict(name='box_scale', prob=1.0, range=[0.8, 1.2]),
            dict(name='box_flip', prob=1.0, axis=['x', 'y'])
        ])
        for data_dict in tqdm(iterable=self.dataset):
            data_dict["gt_boxes"] = data_dict["gt_boxes"][..., :7]
            gt_boxes = data_dict["gt_boxes"].copy()
            points = data_dict["points"].copy()

            if forward_type == 'cuda':
                data_dict["gt_boxes"] = torch.from_numpy(data_dict["gt_boxes"]).cuda()
                data_dict["points"] = torch.from_numpy(data_dict["points"]).cuda()

            aus(data_dict)

            if backward_type == 'cuda':
                if forward_type != 'cuda':
                    data_dict["gt_boxes"] = torch.from_numpy(data_dict["gt_boxes"]).cuda()
                    data_dict["points"] = torch.from_numpy(data_dict["points"]).cuda()
            else:
                if forward_type == 'cuda':
                    data_dict["gt_boxes"] = data_dict["gt_boxes"].cpu().numpy()
                    data_dict["points"] = data_dict["points"].cpu().numpy()

            aus.invert(data_dict)

            if backward_type != 'cuda':
                data_dict["gt_boxes"] = torch.from_numpy(data_dict["gt_boxes"])
                data_dict["points"] = torch.from_numpy(data_dict["points"])
            assert torch.allclose(data_dict["gt_boxes"].cpu(), torch.from_numpy(gt_boxes), rtol=0.01, atol=0.001)
            assert torch.allclose(data_dict["points"].cpu(), torch.from_numpy(points), rtol=0.01, atol=0.001)


if __name__ == '__main__':
    unittest.main()
