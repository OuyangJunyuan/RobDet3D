import unittest
import torch
import open3d
import numpy as np
from rd3d.utils.common_utils import gather
from rd3d.models.backbones_3d.pfe.ops.builder import sampler, querier

visualize = True


def add_masked_color(vis, p1, c1):
    from rd3d.utils.viz_utils import add_points
    if vis:
        point = p1.cpu().detach().numpy()
        color = np.array([c1]).repeat(point.shape[0], axis=0)
        add_points(vis, point, color)


class TestQuerier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xyz, cls.bid, cls.boxes = torch.load("test/data/sampler/xyz_bid_16384x8.cache")
        print(f"xyz: {cls.xyz.shape}")
        print(f"bid: {cls.bid.shape}")
        print(f"box: {cls.boxes.shape}")
        cls.vis = None

    def setUp(self) -> None:
        if visualize:
            self.vis = open3d.visualization.Visualizer()
            self.vis.create_window(window_name=self._testMethodName)
            self.vis.get_render_option().point_size = 1.0
            self.vis.get_render_option().background_color = np.ones(3) * 0.1
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
            self.vis.add_geometry(axis_pcd)
            self.vis.clear_geometries()

    def tearDown(self) -> None:
        if visualize:
            self.vis.run()
            self.vis.destroy_window()

    def test_1_grid_ball_query(self):
        s = sampler.from_cfg(dict(name='havs', sample=4096, voxel=[0.4, 0.4, 0.35], return_hash=True))
        q = querier.from_name('grid_ball')
        indices = s(self.xyz)
        new_xyz = gather(self.xyz, indices)
        voxels = s.return_dict["voxel_sizes"][:, None, 3:6]
        voxel_center = torch.round(self.xyz / voxels) * voxels

        num_queried, indices_query = q(radius=0.20, nsample=32, new_xyz=new_xyz, xyz=self.xyz, **s.return_dict)

        viz_ball_ind_list = list(range(0, 20))
        q_ind = [indices_query[-1, ind, :num_queried[-1, ind]] for ind in viz_ball_ind_list]
        q_ind = torch.cat(q_ind).view(-1)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, q_ind], [0.1, 1, 0.1])
        add_masked_color(self.vis, self.xyz[-1, indices[-1, viz_ball_ind_list]], [1, 0.1, 0.1])

    def test_2_ball_query(self):
        s = sampler.from_cfg(dict(name='havs', sample=4096, voxel=[0.4, 0.4, 0.35]))
        q = querier.from_name('ball')
        indices = s(self.xyz)
        new_xyz = gather(self.xyz, indices)
        num_queried, indices_query = q(radius=0.20, nsample=32, new_xyz=new_xyz, xyz=self.xyz)

        viz_ball_ind_list = list(range(0, 20))
        q_ind = [indices_query[-1, ind, :num_queried[-1, ind]] for ind in viz_ball_ind_list]
        q_ind = torch.cat(q_ind).view(-1)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, q_ind], [0.1, 1, 0.1])
        add_masked_color(self.vis, self.xyz[-1, indices[-1, viz_ball_ind_list]], [1, 0.1, 0.1])


if __name__ == '__main__':
    unittest.main()
