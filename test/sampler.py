import unittest
import torch
import open3d
import numpy as np
from rd3d.models.backbones_3d.pfe.ops.builder import sampler

visualize = False


def add_masked_color(vis, p1, c1):
    from rd3d.utils.viz_utils import add_points
    if vis:
        point = p1.cpu().detach().numpy()
        color = np.array([c1]).repeat(point.shape[0], axis=0)
        add_points(vis, point, color)


class TestSampler(unittest.TestCase):
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

    def test_0_select(self):
        s = sampler.from_cfg(dict(name="select", sample=[0, 4096]))
        indices = s(self.xyz)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, indices[-1]], [1, 0.2, 0.2])

    def test_1_rps(self):
        s = sampler.from_cfg(dict(name="rps", sample=4096))
        indices = s(self.xyz)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, indices[-1]], [1, 0.2, 0.2])

    def test_2_rvs(self):
        s1 = sampler.from_cfg(dict(name="rvs", sample=4096, voxel=[0.4, 0.4, 0.3],
                                   coors_range=[0, -40, -3, 70.4, 40, 1],
                                   channel=3, pool='rand', max_pts_per_voxel=5))
        s2 = sampler.from_cfg(dict(name="rvs", sample=4096, voxel=[0.4, 0.4, 0.3],
                                   coors_range=[0, -40, -3, 70.4, 40, 1],
                                   channel=3, pool='mean', max_pts_per_voxel=5))

        xyz_1 = s1(self.xyz)
        xyz_2 = s2(self.xyz)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, xyz_1[-1], [1, 0.2, 0.2])

    def test_2_rvs_adaptive(self):
        s3 = sampler.from_cfg(dict(name="rvs", sample=4096, voxel=[0.4, 0.4, 0.3],
                                   coors_range=[0, -40, -3, 70.4, 40, 1],
                                   channel=3, pool='mean', max_pts_per_voxel=5, adaptive=True))

        xyz_3 = s3(self.xyz)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, xyz_3[-1], [1, 0.2, 0.2])

    def test_3_d_fps(self):
        s = sampler.from_cfg(dict(name="d-fps", sample=4096))
        indices = s(self.xyz)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, indices[-1]], [1, 0.2, 0.2])

    def test_4_f_fps(self):
        s = sampler.from_cfg(dict(name="f-fps", sample=1024, gamma=1.0))
        mlps = torch.nn.Linear(in_features=3, out_features=32, device=self.xyz.device)
        xyz = self.xyz[:, :4096].contiguous()
        feats = mlps(xyz)
        indices = s(xyz, feats)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, indices[-1]], [1, 0.2, 0.2])

    def test_5_s_fps(self):
        s = sampler.from_cfg(dict(name="s-fps", sample=1024, gamma=1.0, mlps=[32],
                                  train=dict(target={'set_ignore_flag': True, 'extra_width': [1.0, 1.0, 1.0]},
                                             loss={'weight': 0.01, 'tb_tag': 'sasa_1'})
                                  ), input_channels=32)
        s.train()
        s.to(self.xyz.device)
        mlps = torch.nn.Linear(in_features=3, out_features=32, device=self.xyz.device)
        xyz = self.xyz[:, :4096].contiguous()
        feats = mlps(xyz)
        indices = s(xyz, feats)
        batch_dict = dict(gt_boxes=self.boxes)
        s.assign_targets(batch_dict)
        loss = s.get_loss({})
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, indices[-1]], [1, 0.2, 0.2])

    def test_6_ctr(self):
        s = sampler.from_cfg(dict(name='ctr', range=[0, 512], sample=256, mlps=[256],
                                  class_names=['Car', 'Pedestrian', 'Cyclist'],
                                  train=dict(target={'extra_width': [0.5, 0.5, 0.5]},
                                             loss={'weight': 1.0, 'tb_tag': 'sasa_2'})
                                  ), input_channels=32)
        s.train()
        s.to(self.xyz.device)
        mlps = torch.nn.Linear(in_features=3, out_features=32, device=self.xyz.device)
        xyz = self.xyz[:, :4096].contiguous()
        feats = mlps(xyz)
        indices = s(xyz, feats)
        batch_dict = dict(gt_boxes=self.boxes)
        s.assign_targets(batch_dict)
        loss = s.get_loss({})
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, indices[-1]], [1, 0.2, 0.2])

    def test_7_havs(self):
        s = sampler.from_cfg(dict(name='havs', sample=4096, voxel=[0.4, 0.4, 0.35]))
        indices = s(self.xyz)
        add_masked_color(self.vis, self.xyz[-1], [0.2, 0.2, 0.2])
        add_masked_color(self.vis, self.xyz[-1, indices[-1]], [1, 0.2, 0.2])


if __name__ == '__main__':
    unittest.main()
