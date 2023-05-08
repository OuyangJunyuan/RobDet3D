import unittest
import numpy as np
from easydict import EasyDict

from rd3d.utils import viz_utils


class TestInstanceBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = np.random.rand(10000, 4) * 20
        cls.pred_boxes = np.array([[10, 0, 0, 1, 1, 1, 0], [5, 5, 0, 1, 1, 1, 0.75]])
        cls.pred_labels = np.array([1, 2])
        cls.pred_scores = np.array([0.5, 0.8])

    def test0_points(self):
        viz_utils.viz_scene(points=self.points)
        viz_utils.viz_scene(points=(self.points[None, ...]), center=np.array([10, 10, 10]))
        viz_utils.viz_scene(points=(self.points, [0.1, 0.1, 0.1]))
        viz_utils.viz_scene(points=(self.points, [0.1, 0.1, 0.1]))

    def test1_boxes(self):
        viz_utils.viz_scene(boxes=self.pred_boxes)
        viz_utils.viz_scene(boxes=(self.pred_boxes[None, ...]), center=np.array([10, 10, 10]))
        viz_utils.viz_scene(boxes=(self.pred_boxes, [0.1, 0.1, 0.1]))
        viz_utils.viz_scene(boxes=(np.concatenate([self.pred_boxes, self.pred_labels[:, None]], axis=-1)))

    def test2_add_key_points(self):
        viz_utils.viz_scene(key_points=self.points[::200])

    def test3_add_scenes(self):
        viz_utils.viz_scenes((self.points, self.pred_boxes, self.points[::200]),
                             (self.points, self.pred_boxes),
                             offset=[0, 0, 20])


if __name__ == '__main__':
    unittest.main()
